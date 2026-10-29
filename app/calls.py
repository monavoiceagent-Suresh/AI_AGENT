"""Call lifecycle: placing outbound calls and running/persisting connected calls."""

import asyncio
import logging
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import NamedTuple

import asyncpg
import httpx

from app import db, dispatch_db, inbound_db, metrics, prompts
from app.config import Settings
from app.interfaces import LLM, TTS, LLMUsage, Voice
from app.models import BAKBAK_MODELS, TTS_MODEL_LANGUAGES, Agent, AgentIn
from app.phone import normalize_phone
from app.providers.exotel import ExotelClient, ExotelMediaStream, StreamStart
from app.providers.scribe import ScribeRealtime
from app.redaction import mask_number
from app.session import SessionConfig, VoiceSession

log = logging.getLogger(__name__)


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
    asr: ScribeRealtime
    llm: LLM
    tts: TTS
    fallback_tts: TTS | None
    active_calls: int = 0
    # Outbound CallSid -> time placed, holding capacity until the call connects or ends.
    ringing: dict[str, float] = field(default_factory=dict)


def voice_for(agent: AgentIn, settings: Settings) -> Voice:
    if settings.tts_provider == "bakbak":
        default = settings.bakbak_voices[agent.language]
    else:
        default = Voice(voice_id=settings.elevenlabs_tts_voice_id, model_id=settings.elevenlabs_tts_model_id)
    return Voice(voice_id=agent.voice_id or default.voice_id, model_id=agent.tts_model_id or default.model_id)


def fallback_voice_for(agent: AgentIn, settings: Settings) -> Voice | None:
    """ElevenLabs voice used when Bakbak fails; None when disabled or unable to speak the language."""
    if settings.tts_provider == "bakbak" and settings.tts_fallback_enabled:
        model = settings.elevenlabs_tts_model_id
        if agent.language in TTS_MODEL_LANGUAGES.get(model, ()):
            return Voice(voice_id=settings.elevenlabs_tts_voice_id, model_id=model)
    return None


def validate_agent(agent: AgentIn, settings: Settings) -> str | None:
    """Return an error message if the agent's language can't be spoken with its voice and TTS model."""
    if settings.tts_provider == "bakbak":
        if agent.language not in settings.bakbak_voices:
            return f"BAKBAK_VOICES has no voice for language {agent.language!r}"
        if bool(agent.voice_id) != bool(agent.tts_model_id):
            return "Set voice_id and tts_model_id together: a Bakbak voice belongs to exactly one model"
        if agent.tts_model_id and agent.tts_model_id not in BAKBAK_MODELS:
            return f"Unsupported Bakbak model {agent.tts_model_id!r}; use one of {sorted(BAKBAK_MODELS)}"
        return None
    model = voice_for(agent, settings).model_id
    languages = TTS_MODEL_LANGUAGES.get(model)
    if languages is None:
        return f"Unsupported TTS model {model!r}; use one of {sorted(TTS_MODEL_LANGUAGES)}"
    if agent.language not in languages:
        return f"TTS model {model!r} cannot speak language {agent.language!r}; use eleven_v3"
    return None


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


def with_persona(agent: Agent, persona: dict) -> Agent:
    """The agent saying a published persona's text instead of its stored text (voice settings are kept)."""
    return agent.model_copy(update={field: persona[field] for field in dispatch_db.PERSONA_FIELDS})


async def place_outbound_call(
    svc: Services, agent: Agent, number: str, contact_name: str | None, dispatch_lead_id: int | None = None
) -> int:
    settings = svc.settings
    if not settings.enable_outbound_calls:
        raise CallRejected("Outbound calls are disabled (ENABLE_OUTBOUND_CALLS=false)")
    if not capacity_left(svc):
        raise CallRejected("Concurrent call limit reached")
    token = settings.exotel_webhook_secret.get_secret_value()
    callback = f"{settings.public_base_url.rstrip('/')}/exotel/status?token={token}"
    # A bare ten-digit or 0-prefixed number is an Indian one; Exotel needs the country code.
    number = f"+{digits}" if (digits := normalize_phone(number)) else number
    call_sid = await svc.exotel.connect_to_flow(
        number,
        status_callback=callback,
        time_limit=settings.max_call_duration_seconds,
        ring_timeout=RING_TIMEOUT_SECONDS,
    )
    svc.ringing[call_sid] = time.monotonic()
    log.info("outbound call %s placed with agent %s", call_sid, agent.id)
    try:
        return await db.create_outbound_call(
            svc.pool, agent.id, call_sid, settings.exotel_caller_id, number, contact_name, dispatch_lead_id
        )
    except Exception:
        # The callee's phone is already ringing with no row to answer it from: make that findable.
        log.exception("outbound call %s was placed but could not be recorded", call_sid)
        raise


async def call_status(svc: Services, call_sid: str, status: str) -> None:
    svc.ringing.pop(call_sid, None)
    status = status.replace("-", "_")  # no-answer -> no_answer
    if await db.apply_status_callback(svc.pool, call_sid, status):
        log.info("outbound call %s ended before connecting: %s", call_sid, status)


class Resolved(NamedTuple):
    agent: Agent
    call_id: int | None
    caller_context: str
    # The stream's start, with `called_number` set to the number the call was routed by.
    start: StreamStart


async def resolve_agent(svc: Services, start: StreamStart) -> Resolved:
    """Pick the agent and caller context for a new media stream: the outbound call's agent, else the inbound default."""
    call = await db.get_call_by_sid(svc.pool, start.call_sid)
    if call is not None:
        if call["status"] != "initiated":
            raise CallRejected(f"Call {start.call_sid} was already handled")
        svc.ringing.pop(start.call_sid, None)
        agent = Agent(**dict(await db.get_agent(svc.pool, call["agent_id"])))
        if call["dispatch_lead_id"] is not None:
            # A campaign call says what the persona said when the campaign was started.
            persona = await dispatch_db.get_job_persona(svc.pool, call["dispatch_lead_id"])
            if persona:
                agent = with_persona(agent, persona)
        return Resolved(agent, call["id"], prompts.caller_context("outbound", call["contact_name"]), start)
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
) -> None:
    settings = svc.settings
    config = session_config(agent, settings, caller_context)
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
    try:
        async with svc.asr.connect(language=agent.language, sample_rate=media.sample_rate) as transcriber:
            session = VoiceSession(
                config, media=media, asr=transcriber, llm=svc.llm, tts=svc.tts, fallback_tts=svc.fallback_tts
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
            _finalize(svc, agent, config, session, call_id, end_reason, started_at, connected_seconds)
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
) -> None:
    settings = svc.settings
    turns = session.turns if session else []
    outcome, fields, summary, extraction_usage = await extract_outcome(svc, agent, config.llm_model, turns)
    llm_usage = (session.llm_usage if session else LLMUsage(0, 0, 0.0)) + extraction_usage
    usage = {
        "asr_seconds": round(session.asr_seconds, 2) if session else 0.0,
        "tts_provider": settings.tts_provider,
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
                metrics.tts_inr_per_1k_chars(settings, settings.tts_provider, config.voice.model_id),
            ),
            (
                usage["fallback_tts_characters"],
                metrics.tts_inr_per_1k_chars(
                    settings, "elevenlabs", config.fallback_voice.model_id if config.fallback_voice else ""
                ),
            ),
        ],
        llm_cost_usd=usage["llm_cost_usd"],
    )
    await db.finish_call(
        svc.pool,
        call_id,
        end_reason=end_reason,
        ended_at=started_at + timedelta(seconds=connected_seconds),
        connected_seconds=round(connected_seconds, 2),
        usage=usage,
        cost=cost,
        latency=metrics.latency_summary(turns),
        turns=turns,
        outcome=outcome,
        outcome_data=fields,
        summary=summary,
        report_url=settings.inbound_report_url,
    )
    log.info("call %s ended (%s) after %.1fs, est. cost INR %s", call_id, end_reason, connected_seconds, cost["total"])


async def extract_outcome(
    svc: Services, agent: Agent, model: str, turns: list[dict]
) -> tuple[str, dict, str, LLMUsage]:
    no_usage = LLMUsage(0, 0, 0.0)
    if not any(t["user_text"] for t in turns):
        return "no_conversation", {}, "The caller did not speak.", no_usage
    transcript: list[tuple[str, str]] = []
    for turn in turns:
        if turn["user_text"]:
            transcript.append(("Caller", turn["user_text"]))
        transcript.append(("Agent", turn["agent_text"]))
    try:
        raw, usage = await svc.llm.complete(prompts.outcome_messages(agent, transcript), model)
    except Exception:
        log.exception("outcome extraction failed")
        return "extraction_failed", {}, "", no_usage
    return (*prompts.parse_outcome(agent, raw), usage)
