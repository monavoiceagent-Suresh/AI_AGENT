"""Call lifecycle: placing outbound calls and running/persisting connected calls."""

import asyncio
import logging
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import NamedTuple

import asyncpg
import httpx

from app import db, dispatch_db, inbound_db, metrics, prompts
from app.config import ASRProvider, Settings, TTSProvider
from app.interfaces import ASR, LLM, TTS, LLMUsage, Transcriber, Voice
from app.models import BAKBAK_MODELS, TTS_MODEL_LANGUAGES, Agent, AgentIn, OutcomeRuleIn
from app.phone import normalize_phone
from app.providers.budget import BudgetedLLM
from app.providers.exotel import ExotelClient, ExotelMediaStream, StreamStart
from app.redaction import mask_number
from app.session import SessionConfig, VoiceSession

log = logging.getLogger(__name__)

# Languages each TTS provider can speak at all, independent of the specific ElevenLabs model tier
# (that finer-grained check stays in TTS_MODEL_LANGUAGES). Verified: ElevenLabs model pages for the
# premium/standard tiers; Deepgram Aura live (2026-10-31) and its own model list, which has no hi/ta/te voices.
TTS_PROVIDER_LANGUAGES: dict[TTSProvider, set[str] | None] = {
    "elevenlabs_premium": {"en", "hi", "ta", "te"},
    "elevenlabs_standard": {"en", "hi", "ta"},
    "deepgram_aura": {"en"},
    "bakbak": None,  # dynamic: whichever languages BAKBAK_VOICES configures
}


# Exotel rings the callee for this long; an unanswered reservation expires shortly after.
RING_TIMEOUT_SECONDS = 45
RESERVATION_SECONDS = 90
# Pause before the second read of Exotel's record of a call that has only just connected.
LOOKUP_RETRY_SECONDS = 0.5


class CallRejected(Exception):
    pass


@dataclass
class Services:
    settings: Settings
    pool: asyncpg.Pool
    http: httpx.AsyncClient
    exotel: ExotelClient
    asr: ASR
    fallback_asr: ASR | None
    # Every ASR/TTS provider this deployment actually has credentials for (None where it doesn't), so a
    # per-agent provider override can be honoured without rebuilding anything at call time. `asr`/`tts` and
    # `fallback_asr`/`fallback_tts` above are always these same maps' entries for the global default provider.
    asr_by_provider: dict[ASRProvider, "ASR | None"]
    llm: LLM
    tts: TTS
    fallback_tts: TTS | None
    tts_by_provider: dict[TTSProvider, "TTS | None"]
    active_calls: int = 0
    # Outbound CallSid -> time placed, holding capacity until the call connects or ends.
    ringing: dict[str, float] = field(default_factory=dict)


def _elevenlabs_model_id(provider: TTSProvider, settings: Settings) -> str:
    return (
        settings.elevenlabs_premium_model_id if provider == "elevenlabs_premium" else settings.elevenlabs_tts_model_id
    )


def _default_tts_voice(provider: TTSProvider, settings: Settings, language: str) -> Voice | None:
    """This provider's voice for `language` from settings alone (no agent-level override); None if it can't
    speak that language at all, or (bakbak) has no voice configured for it."""
    if provider == "bakbak":
        return settings.bakbak_voices.get(language)
    if language not in (TTS_PROVIDER_LANGUAGES[provider] or ()):
        return None
    if provider == "deepgram_aura":
        return Voice(voice_id=settings.deepgram_tts_voice, model_id="aura")
    if not settings.elevenlabs_tts_voice_id:
        return None
    return Voice(voice_id=settings.elevenlabs_tts_voice_id, model_id=_elevenlabs_model_id(provider, settings))


def voice_for(agent: AgentIn, settings: Settings) -> Voice:
    """The primary provider's voice for this agent: its own voice_id/tts_model_id override the default, if set."""
    provider = agent.tts_provider or settings.tts_provider
    default = _default_tts_voice(provider, settings, agent.language)
    assert default is not None, "validate_agent must reject an agent whose language the primary cannot speak"
    return Voice(voice_id=agent.voice_id or default.voice_id, model_id=agent.tts_model_id or default.model_id)


def fallback_voice_for(agent: AgentIn, settings: Settings) -> Voice | None:
    """The fallback provider's voice, used when the primary fails before producing audio. Always the default
    voice for that provider: an agent's voice_id override is specific to the PRIMARY provider's identifier
    namespace and must not be reused for a different (fallback) provider. None when there is no fallback
    provider, or it cannot speak this agent's language."""
    provider = agent.tts_fallback_provider or settings.tts_fallback_provider
    if provider is None:
        return None
    return _default_tts_voice(provider, settings, agent.language)


def validate_agent(agent: AgentIn, settings: Settings) -> str | None:
    """Return an error message if the agent's language can't be spoken by the primary TTS provider."""
    provider = agent.tts_provider or settings.tts_provider
    if provider == "bakbak":
        if agent.language not in settings.bakbak_voices:
            return f"BAKBAK_VOICES has no voice for language {agent.language!r}"
        if bool(agent.voice_id) != bool(agent.tts_model_id):
            return "Set voice_id and tts_model_id together: a Bakbak voice belongs to exactly one model"
        if agent.tts_model_id and agent.tts_model_id not in BAKBAK_MODELS:
            return f"Unsupported Bakbak model {agent.tts_model_id!r}; use one of {sorted(BAKBAK_MODELS)}"
        return None
    if agent.language not in (TTS_PROVIDER_LANGUAGES[provider] or ()):
        return f"TTS provider {provider!r} cannot speak language {agent.language!r}"
    if provider == "deepgram_aura":
        return None  # one fixed English voice; no per-agent model/voice choice to validate
    model = voice_for(agent, settings).model_id
    languages = TTS_MODEL_LANGUAGES.get(model)
    if languages is None:
        return f"Unsupported TTS model {model!r}; use one of {sorted(TTS_MODEL_LANGUAGES)}"
    if agent.language not in languages:
        return f"TTS model {model!r} cannot speak language {agent.language!r}; use eleven_v3"
    return None


@asynccontextmanager
async def connect_asr(
    asr: ASR, fallback_asr: "ASR | None", *, language: str, sample_rate: int
) -> AsyncIterator[Transcriber]:
    """Connect with `asr`; if the connection itself cannot be established, try `fallback_asr` once.

    This is a connect-time fallback only. Once a session is open, a later failure is never retried on the
    other provider: doing so after audio has already been transcribed could duplicate or drop turns, so that
    kind of mid-call failure still ends the call today, exactly as it did with a single ASR provider.
    """
    primary_ctx = asr.connect(language=language, sample_rate=sample_rate)
    try:
        transcriber = await primary_ctx.__aenter__()
    except Exception:
        if fallback_asr is None:
            raise
        log.warning("primary ASR failed to connect; using the fallback", exc_info=True)
        async with fallback_asr.connect(language=language, sample_rate=sample_rate) as fallback_transcriber:
            yield fallback_transcriber
        return
    try:
        yield transcriber
    except BaseException:
        # Give the primary's own context manager a chance to see the real failure (and close itself
        # accordingly); if it doesn't suppress it (none of our adapters do), it keeps propagating. Either
        # way, this is a session-time failure, so it is never retried on the fallback.
        if not await primary_ctx.__aexit__(*sys.exc_info()):
            raise
    else:
        await primary_ctx.__aexit__(None, None, None)


def resolve_tts(svc: "Services", agent: AgentIn) -> tuple[TTS, str, TTS | None, str | None]:
    """The TTS instances to use for this call: (primary, its provider name, fallback or None, its name or
    None). An agent's tts_provider/tts_fallback_provider overrides which provider is used, but only if this
    engine deployment actually has credentials for it; otherwise the global default is used instead (primary)
    or the fallback is simply disabled for this call (fallback), and a warning is logged either way, so a
    workspace's choice can never silently break a call.
    """
    settings = svc.settings
    name = agent.tts_provider or settings.tts_provider
    instance = svc.tts if name == settings.tts_provider else svc.tts_by_provider.get(name)
    if instance is None:
        log.warning(
            "agent's tts_provider override %r has no credentials configured; using %r", name, settings.tts_provider
        )
        name, instance = settings.tts_provider, svc.tts
    fallback_name = agent.tts_fallback_provider or settings.tts_fallback_provider
    fallback_instance = None
    if fallback_name is not None:
        fallback_instance = (
            svc.fallback_tts
            if fallback_name == settings.tts_fallback_provider
            else svc.tts_by_provider.get(fallback_name)
        )
        if fallback_instance is None:
            log.warning(
                "tts_fallback_provider %r has no credentials configured; no TTS fallback for this call", fallback_name
            )
            fallback_name = None
    return instance, name, fallback_instance, fallback_name


def resolve_asr(svc: "Services", agent: AgentIn) -> tuple[ASR, str, "ASR | None", str | None]:
    """Same resolution as resolve_tts, for ASR."""
    settings = svc.settings
    name = agent.asr_provider or settings.asr_provider
    instance = svc.asr if name == settings.asr_provider else svc.asr_by_provider.get(name)
    if instance is None:
        log.warning(
            "agent's asr_provider override %r has no credentials configured; using %r", name, settings.asr_provider
        )
        name, instance = settings.asr_provider, svc.asr
    fallback_name = agent.asr_fallback_provider or settings.asr_fallback_provider
    fallback_instance = None
    if fallback_name is not None:
        fallback_instance = (
            svc.fallback_asr
            if fallback_name == settings.asr_fallback_provider
            else svc.asr_by_provider.get(fallback_name)
        )
        if fallback_instance is None:
            log.warning(
                "asr_fallback_provider %r has no credentials configured; no ASR fallback for this call", fallback_name
            )
            fallback_name = None
    return instance, name, fallback_instance, fallback_name


def session_config(agent: Agent, settings: Settings, caller_context: str) -> SessionConfig:
    return SessionConfig(
        system_prompt=prompts.system_prompt(agent, caller_context),
        greeting=agent.greeting,
        fallback=prompts.FALLBACK_PHRASES[agent.language],
        language=agent.language,
        voice=voice_for(agent, settings),
        fallback_voice=fallback_voice_for(agent, settings),
        llm_model=agent.llm_model or settings.primary_llm_model,
        barge_in=settings.barge_in_enabled,
        idle_timeout_seconds=settings.session_idle_timeout_seconds,
        max_duration_seconds=settings.max_call_duration_seconds,
        log_transcripts=settings.log_transcripts,
        llm_dedup_window_ms=settings.llm_dedup_window_ms,
        max_consecutive_llm_failures=settings.max_consecutive_llm_failures,
    )


def capacity_left(svc: Services, *, inbound: bool = False) -> bool:
    """Whether one more call fits. Outbound and campaign calls leave INBOUND_RESERVED_CALLS slots free."""
    now = time.monotonic()
    for call_sid, placed in list(svc.ringing.items()):
        if now - placed > RESERVATION_SECONDS:
            del svc.ringing[call_sid]
    settings = svc.settings
    limit = settings.max_concurrent_calls - (0 if inbound else settings.inbound_reserved_calls)
    return svc.active_calls + len(svc.ringing) < limit


def with_call_voice(agent: Agent, voice_id: str | None, tts_model_id: str | None) -> Agent:
    """The voice this call was placed with, on its own copy of the agent; the agent's voice when none was chosen."""
    if not (voice_id and tts_model_id):
        return agent
    return agent.model_copy(update={"voice_id": voice_id, "tts_model_id": tts_model_id})


def with_persona(agent: Agent, persona: dict) -> Agent:
    """The agent saying a published persona's text instead of its stored text (voice settings are kept).

    `persona` is JSON loaded straight from storage. A route or job published before outcome_rules/
    outcome_fallback existed has neither key at all, hence .get() with the same defaults PersonaIn itself
    uses. model_copy does not re-validate, so outcome_rules (plain dicts once loaded from JSON) is parsed
    back into OutcomeRuleIn here rather than leaving the agent with raw dicts where prompts.py expects
    .name/.definition attributes.
    """
    text_fields = [f for f in dispatch_db.PERSONA_FIELDS if f not in ("outcome_rules", "outcome_fallback")]
    update = {
        field: (persona.get(field) if field in dispatch_db.PERSONA_OPTIONAL_FIELDS else persona[field])
        for field in text_fields
    }
    update["outcome_rules"] = [OutcomeRuleIn(**r) for r in persona.get("outcome_rules", [])]
    update["outcome_fallback"] = persona.get("outcome_fallback")
    return agent.model_copy(update=update)


async def place_outbound_call(
    svc: Services,
    agent: Agent,
    number: str,
    contact_name: str | None,
    dispatch_lead_id: int | None = None,
    *,
    voice_id: str | None = None,
    tts_model_id: str | None = None,
) -> int:
    settings = svc.settings
    if not settings.enable_outbound_calls:
        raise CallRejected("Outbound calls are disabled (ENABLE_OUTBOUND_CALLS=false)")
    if not capacity_left(svc):
        raise CallRejected("Concurrent call limit reached")
    # Reserve before the first await, including concurrent HTTP test/management requests.
    reservation = f"placing-{uuid.uuid4()}"
    svc.ringing[reservation] = time.monotonic()
    token = settings.exotel_webhook_secret.get_secret_value()
    callback = f"{settings.public_base_url.rstrip('/')}/exotel/status?token={token}"
    # A bare ten-digit or 0-prefixed number is an Indian one; Exotel needs the country code.
    number = f"+{digits}" if (digits := normalize_phone(number)) else number
    try:
        call_sid = await svc.exotel.connect_to_flow(
            number,
            status_callback=callback,
            time_limit=settings.max_call_duration_seconds,
            ring_timeout=RING_TIMEOUT_SECONDS,
        )
    finally:
        svc.ringing.pop(reservation, None)
    svc.ringing[call_sid] = time.monotonic()
    log.info("outbound call %s placed with agent %s", call_sid, agent.id)
    try:
        return await db.create_outbound_call(
            svc.pool,
            agent.id,
            call_sid,
            settings.exotel_caller_id,
            number,
            contact_name,
            dispatch_lead_id,
            voice_id=voice_id,
            tts_model_id=tts_model_id,
        )
    except Exception:
        # The callee's phone is already ringing with no row to answer it from: make that findable.
        log.exception("outbound call %s was placed but could not be recorded", call_sid)
        raise


async def call_status(svc: Services, call_sid: str, status: str) -> None:
    status = status.replace("-", "_")  # no-answer -> no_answer
    if status not in {"completed", "failed", "busy", "no_answer", "canceled", "cancelled"}:
        return
    svc.ringing.pop(call_sid, None)
    if await db.apply_status_callback(svc.pool, call_sid, status):
        log.info("outbound call %s ended before connecting: %s", call_sid, status)


class Resolved(NamedTuple):
    agent: Agent
    call_id: int | None
    caller_context: str
    # The stream's start, with `called_number` set to the number the call was routed by.
    start: StreamStart
    campaign_id: int | None = None


async def resolve_agent(svc: Services, start: StreamStart) -> Resolved:
    """Pick the agent and caller context for a new media stream: the outbound call's agent, else the inbound default."""
    call = await db.get_call_by_sid(svc.pool, start.call_sid)
    if call is not None:
        if call["status"] != "initiated":
            raise CallRejected(f"Call {start.call_sid} was already handled")
        svc.ringing.pop(start.call_sid, None)
        agent = Agent(**dict(await db.get_agent(svc.pool, call["agent_id"])))
        campaign_id = None
        if call["dispatch_lead_id"] is not None:
            # A campaign call says what the persona said when the campaign was started.
            job = await dispatch_db.get_job_runtime(svc.pool, call["dispatch_lead_id"])
            if job is not None:
                campaign_id = job["campaign_id"]
                if job["persona"]:
                    agent = with_persona(agent, job["persona"])
        agent = with_call_voice(agent, call["voice_id"], call["tts_model_id"])
        return Resolved(agent, call["id"], prompts.caller_context("outbound", call["contact_name"]), start, campaign_id)
    return await _resolve_inbound(svc, start)


async def _exophone_from_exotel(svc: Services, start: StreamStart) -> str | None:
    """The ExoPhone that Exotel's own record of this call says was dialled, as digits, or None.

    The stream's `to` is not always the dialled ExoPhone (Exotel documents the field's shape but not its meaning), so
    when it matches no route this reads the call's record (read-only, the account's credentials). Any failure
    leaves the call to be refused exactly as before; it can never break call setup.
    """
    for attempt in (1, 2):
        try:
            record = await svc.exotel.get_call_record(
                start.call_sid, wait_seconds=svc.settings.exotel_lookup_timeout_seconds
            )
        except Exception as exc:  # network, authentication, a record that is not there yet
            log.warning("could not read Exotel's record of call %s (attempt %d): %s", start.call_sid, attempt, exc)
        else:
            log.info(
                "exotel call record for %s: direction=%r to=%r phone_number_sid=%r forwarded_from=%s",
                start.call_sid,
                record.direction,
                record.to,
                record.phone_number_sid,
                mask_number(record.forwarded_from) if record.forwarded_from else "-",
            )
            if record.direction and not record.direction.lower().startswith("inbound"):
                return None  # a definitive answer: not an inbound call
            if exophone := normalize_phone(record.phone_number_sid):
                return exophone
        if attempt == 1:
            # A call that has only just connected may not be in Exotel's records yet: one more try, briefly later.
            await asyncio.sleep(LOOKUP_RETRY_SECONDS)
    return None


async def _resolve_inbound(svc: Services, start: StreamStart) -> Resolved:
    """Answer a call that no outbound call row matches, by the number that was dialled.

    The route is looked up by the stream's `to`; only if that matches nothing is the ExoPhone in Exotel's call record
    tried. A published route answers with its own persona. A number with no route is refused, unless the engine-wide
    default agent (which belongs to no workspace) is still enabled. A workspace's persona is never used for a
    number it does not own.
    """
    settings = svc.settings
    if not settings.enable_inbound_calls:
        raise CallRejected("Inbound calls are disabled (ENABLE_INBOUND_CALLS=false)")
    dialled = normalize_phone(start.to_number)
    route = await inbound_db.find_route_for_call(svc.pool, dialled) if dialled else None
    found_by = "the stream's to"
    exophone: str | None = None
    if route is None and settings.exotel_resolve_called_number:
        exophone = await _exophone_from_exotel(svc, start)
        if exophone and exophone != dialled:
            route = await inbound_db.find_route_for_call(svc.pool, exophone)
            found_by = "the ExoPhone in Exotel's call record"
    if route is not None:
        if not route["active"]:
            raise CallRejected(f"Inbound number {route['phone']} is switched off")
        base = Agent(**dict(await db.get_agent(svc.pool, route["agent_id"])))
        agent = with_persona(base, route["persona"])
        routed_by = route["phone"]
        log.info(
            "inbound call %s (stream to %r): route %s v%s found by %s answers with persona %r (rev %s) on agent %s",
            start.call_sid,
            start.to_number,
            routed_by,
            route["version"],
            found_by,
            route["persona"]["key"],
            route["persona"]["version"],
            agent.id,
        )
    else:
        record = await db.get_inbound_default_agent(svc.pool) if settings.inbound_legacy_default_enabled else None
        answered_by = "default_agent" if record is not None else "refused"
        routed_by = exophone or dialled
        log.warning(
            "inbound call %s: stream to %r (read as %s), Exotel ExoPhone %s, matches no published route; %s",
            start.call_sid,
            start.to_number,
            dialled,
            exophone or "not found",
            f"answering with the engine-wide default agent {record['id']}"
            if record is not None
            else "refusing the call (no default agent is enabled)",
        )
        try:
            seen = (
                start.to_number if not exophone or exophone == dialled else f"{start.to_number} / ExoPhone {exophone}"
            )
            await inbound_db.note_unrouted(svc.pool, routed_by, seen, answered_by)
        except Exception:
            log.exception("could not record the unrouted number %r", start.to_number)
        if record is None:
            raise CallRejected(f"No inbound route is published for the dialled number {start.to_number!r}")
        agent = Agent(**dict(record))
    if not capacity_left(svc, inbound=True):
        raise CallRejected("Concurrent call limit reached")
    # Hold the slot until run_call counts the call, so a campaign cannot take it in between.
    svc.ringing[start.call_sid] = time.monotonic()
    return Resolved(agent, None, prompts.caller_context("inbound", None), replace(start, called_number=routed_by))


async def run_call(
    svc: Services,
    media: ExotelMediaStream,
    start: StreamStart,
    agent: Agent,
    call_id: int | None,
    caller_context: str,
    campaign_id: int | None = None,
) -> None:
    settings = svc.settings
    config = session_config(agent, settings, caller_context)
    tts, tts_provider_name, fallback_tts, fallback_tts_provider_name = resolve_tts(svc, agent)
    asr, _, fallback_asr, _ = resolve_asr(svc, agent)
    started_at = datetime.now(UTC)
    started = time.monotonic()
    call_id = await db.start_call(
        svc.pool,
        call_id=call_id,
        agent_id=agent.id,
        call_sid=start.call_sid,
        stream_sid=start.stream_sid,
        from_number=start.from_number,
        to_number=start.to_number,
        dialled_number=start.called_number,
        started_at=started_at,
    )
    log.info("call %s connected (agent %s, %s Hz)", call_id, agent.id, media.sample_rate)
    session: VoiceSession | None = None
    cancelled = False
    svc.ringing.pop(start.call_sid, None)  # the slot held while this call was being set up
    svc.active_calls += 1
    # Fresh per call: it only caps this one call's own requests, on top of each provider's own shared throttle.
    call_llm = BudgetedLLM(svc.llm, max_requests=settings.llm_max_requests_per_call, call_sid=start.call_sid)
    try:
        async with connect_asr(
            asr, fallback_asr, language=agent.language, sample_rate=media.sample_rate
        ) as transcriber:
            session = VoiceSession(
                config, media=media, asr=transcriber, llm=call_llm, tts=tts, fallback_tts=fallback_tts
            )
            end_reason = await session.run()
    except asyncio.CancelledError:
        end_reason, cancelled = "server_cancelled", True
    except Exception as exc:
        log.exception("call %s failed", call_id)
        end_reason = f"error:{type(exc).__name__}"
    finally:
        svc.active_calls -= 1
        await media.close()

    # Persist in a separate task so a cancelled handler cannot abort the write half-way.
    connected_seconds = time.monotonic() - started
    await asyncio.shield(
        asyncio.ensure_future(
            _finalize(
                svc,
                agent,
                config,
                session,
                call_id,
                end_reason,
                started_at,
                connected_seconds,
                call_llm,
                campaign_id,
                tts_provider_name,
                fallback_tts_provider_name,
            )
        )
    )
    if cancelled:
        raise asyncio.CancelledError


async def _finalize(
    svc: Services,
    agent: Agent,
    config: SessionConfig,
    session: VoiceSession | None,
    call_id: int,
    end_reason: str,
    started_at: datetime,
    connected_seconds: float,
    llm: LLM,
    campaign_id: int | None,
    tts_provider_name: str,
    fallback_tts_provider_name: str | None,
) -> None:
    settings = svc.settings
    turns = session.turns if session else []
    # The same per-call budget as the conversation: outcome extraction is one more LLM request, not a free one.
    outcome, fields, summary, extraction_usage = await extract_outcome(llm, agent, config.llm_model, turns)
    llm_usage = (session.llm_usage if session else LLMUsage(0, 0, 0.0)) + extraction_usage
    usage = {
        "asr_seconds": round(session.asr_seconds, 2) if session else 0.0,
        "tts_provider": tts_provider_name,
        "tts_model_id": config.voice.model_id,
        "tts_characters": session.tts_characters if session else 0,
        "fallback_tts_characters": session.fallback_tts_characters if session else 0,
        "llm_model": config.llm_model,
        "llm_prompt_tokens": llm_usage.prompt_tokens,
        "llm_completion_tokens": llm_usage.completion_tokens,
        "llm_cost_usd": round(llm_usage.cost_usd, 6),
    }
    cost = metrics.call_cost(
        settings,
        connected_seconds=connected_seconds,
        asr_seconds=usage["asr_seconds"],
        tts_usage=[
            (
                usage["tts_characters"],
                metrics.tts_inr_per_1k_chars(settings, tts_provider_name, config.voice.model_id),
            ),
            (
                usage["fallback_tts_characters"],
                metrics.tts_inr_per_1k_chars(
                    settings,
                    fallback_tts_provider_name or "",
                    config.fallback_voice.model_id if config.fallback_voice else "",
                ),
            ),
        ],
        llm_cost_usd=usage["llm_cost_usd"],
    )
    latency = metrics.latency_summary(turns)
    await db.finish_call(
        svc.pool,
        call_id,
        end_reason=end_reason,
        ended_at=started_at + timedelta(seconds=connected_seconds),
        connected_seconds=round(connected_seconds, 2),
        usage=usage,
        cost=cost,
        latency=latency,
        turns=turns,
        outcome=outcome,
        outcome_data=fields,
        summary=summary,
        report_url=settings.inbound_report_url,
    )
    log.info(
        "call %s ended: campaign=%s persona=%s voice=%s/%s end=%s duration=%.1fs response_p50_ms=%s outcome=%s "
        "cost_inr=%s",
        call_id,
        campaign_id,
        agent.persona_key or "-",
        config.voice.voice_id,
        config.voice.model_id,
        end_reason,
        connected_seconds,
        latency["response_ms_p50"],
        outcome,
        cost["total"],
    )


async def extract_outcome(llm: LLM, agent: Agent, model: str, turns: list[dict]) -> tuple[str, dict, str, LLMUsage]:
    no_usage = LLMUsage(0, 0, 0.0)
    if not any(t["user_text"] for t in turns):
        return "no_conversation", {}, "The caller did not speak.", no_usage
    transcript: list[tuple[str, str]] = []
    for turn in turns:
        if turn["user_text"]:
            transcript.append(("Caller", turn["user_text"]))
        transcript.append(("Agent", turn["agent_text"]))
    try:
        raw, usage = await llm.complete(prompts.outcome_messages(agent, transcript), model)
    except Exception:
        log.exception("outcome extraction failed")
        return "extraction_failed", {}, "", no_usage
    return (*prompts.parse_outcome(agent, raw), usage)
