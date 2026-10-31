import asyncio
import logging
from contextlib import asynccontextmanager, suppress

import httpx
from fastapi import FastAPI

from app import api, db, dispatch
from app.calls import Services
from app.config import get_settings
from app.providers.bakbak_tts import BakbakTTS
from app.providers.chat_completions import ChatCompletionsLLM
from app.providers.deepgram_stt import DeepgramSTT
from app.providers.deepgram_tts import DeepgramTTS
from app.providers.elevenlabs_tts import ElevenLabsTTS
from app.providers.exotel import ExotelClient
from app.providers.failover import FailoverLLM
from app.providers.rate_limit import TokenBucket
from app.providers.scribe import ScribeRealtime

log = logging.getLogger("app")

# Timeout and retries for a provider that does not set its own: a primary must give up quickly so that the fallback
# can answer (every retry adds a full timeout first); a fallback has nothing behind it.
PRIMARY_TIMEOUT_MS, PRIMARY_RETRIES = 4000, 0
FALLBACK_TIMEOUT_MS, FALLBACK_RETRIES = 8000, 1


def _llm_client(http: httpx.AsyncClient, settings, provider: str, *, primary: bool) -> ChatCompletionsLLM:
    """One client per provider, created once at startup and shared by every call: its rate limiter and 429
    cooldown are therefore a real process-wide budget, not just a per-call one."""
    timeout, retries = (PRIMARY_TIMEOUT_MS, PRIMARY_RETRIES) if primary else (FALLBACK_TIMEOUT_MS, FALLBACK_RETRIES)
    backoff_kwargs = {
        "retry_backoff_base_ms": settings.llm_retry_backoff_base_ms,
        "retry_backoff_max_ms": settings.llm_retry_backoff_max_ms,
    }
    if provider == "groq":
        return ChatCompletionsLLM(
            http,
            name="Groq",
            api_key=settings.groq_api_key.get_secret_value(),
            base_url=settings.groq_base_url,
            max_tokens=settings.groq_max_output_tokens,
            temperature=settings.groq_temperature,
            timeout_ms=settings.groq_timeout_ms or timeout,
            max_retries=retries if settings.groq_max_retries is None else settings.groq_max_retries,
            limiter=TokenBucket(settings.groq_max_requests_per_minute),
            **backoff_kwargs,
        )
    return ChatCompletionsLLM(
        http,
        name="OpenRouter",
        api_key=settings.openrouter_api_key.get_secret_value(),
        base_url=settings.openrouter_base_url,
        max_tokens=settings.openrouter_max_output_tokens,
        temperature=settings.openrouter_temperature,
        timeout_ms=settings.openrouter_timeout_ms or timeout,
        max_retries=retries if settings.openrouter_max_retries is None else settings.openrouter_max_retries,
        limiter=TokenBucket(settings.openrouter_max_requests_per_minute),
        **backoff_kwargs,
    )


def build_tts_providers(http: httpx.AsyncClient, settings) -> dict:
    """One instance per TTS provider whose credentials are present; a provider selected (primary or fallback)
    with no credentials simply has no instance, same as how the LLM fallback already behaves below."""
    elevenlabs = (
        ElevenLabsTTS(
            http,
            api_key=settings.elevenlabs_api_key.get_secret_value(),
            stability=settings.elevenlabs_tts_stability,
            similarity_boost=settings.elevenlabs_tts_similarity_boost,
        )
        if settings.elevenlabs_tts_voice_id
        else None
    )
    bakbak = (
        BakbakTTS(http, api_key=settings.bakbak_api_key.get_secret_value(), base_url=settings.bakbak_base_url)
        if settings.bakbak_api_key
        else None
    )
    deepgram = DeepgramTTS(api_key=settings.deepgram_api_key.get_secret_value()) if settings.deepgram_api_key else None
    return {
        "elevenlabs_premium": elevenlabs,
        "elevenlabs_standard": elevenlabs,
        "bakbak": bakbak,
        "deepgram_aura": deepgram,
    }


def build_asr_providers(settings) -> dict:
    scribe = ScribeRealtime(
        api_key=settings.elevenlabs_api_key.get_secret_value(),
        model_id=settings.elevenlabs_asr_model_id,
        silence_ms=settings.endpointing_min_silence_ms,
        connect_timeout_ms=settings.asr_timeout_ms,
    )
    deepgram = (
        DeepgramSTT(
            api_key=settings.deepgram_api_key.get_secret_value(),
            silence_ms=settings.endpointing_min_silence_ms,
            connect_timeout_ms=settings.asr_timeout_ms,
        )
        if settings.deepgram_api_key
        else None
    )
    return {"elevenlabs_scribe": scribe, "deepgram_nova3": deepgram}


def build_llm(http: httpx.AsyncClient, settings) -> FailoverLLM:
    """LLM_PROVIDER answers first; the other provider is the automatic fallback when its key is set."""
    primary_name = settings.llm_provider
    fallback_name = "openrouter" if primary_name == "groq" else "groq"
    fallback_key = settings.openrouter_api_key if fallback_name == "openrouter" else settings.groq_api_key
    fallback = _llm_client(http, settings, fallback_name, primary=False) if fallback_key is not None else None
    fallback_model = settings.openrouter_model if fallback_name == "openrouter" else settings.groq_model
    log.info("LLM primary: %s; fallback: %s", primary_name, fallback_name if fallback else "none")
    return FailoverLLM(
        primary=_llm_client(http, settings, primary_name, primary=True),
        fallback=fallback,
        fallback_model=fallback_model or "",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pool = await db.create_pool(settings.database_url, settings.database_pool_min_size, settings.database_pool_max_size)
    if applied := await db.migrate(pool):
        log.info("applied migrations: %s", ", ".join(applied))
    # One pooled client keeps TLS connections warm to Exotel, ElevenLabs, Deepgram, Groq and OpenRouter.
    http = httpx.AsyncClient(timeout=settings.tts_timeout_ms / 1000)
    tts_by_provider = build_tts_providers(http, settings)
    asr_by_provider = build_asr_providers(settings)
    primary_tts = tts_by_provider[settings.tts_provider]
    assert primary_tts is not None, "Settings validation guarantees the primary TTS provider has credentials"
    fallback_tts = tts_by_provider[settings.tts_fallback_provider] if settings.tts_fallback_provider else None
    if settings.tts_fallback_provider and fallback_tts is None:
        log.warning(
            "TTS_FALLBACK_PROVIDER=%s has no credentials configured; no TTS fallback", settings.tts_fallback_provider
        )
    primary_asr = asr_by_provider[settings.asr_provider]
    assert primary_asr is not None, "Settings validation guarantees the primary ASR provider has credentials"
    fallback_asr = asr_by_provider[settings.asr_fallback_provider] if settings.asr_fallback_provider else None
    if settings.asr_fallback_provider and fallback_asr is None:
        log.warning(
            "ASR_FALLBACK_PROVIDER=%s has no credentials configured; no ASR fallback", settings.asr_fallback_provider
        )
    llm = build_llm(http, settings)
    app.state.services = Services(
        settings=settings,
        pool=pool,
        http=http,
        exotel=ExotelClient(
            http,
            base_url=settings.exotel_api_base_url,
            account_sid=settings.exotel_account_sid,
            api_key=settings.exotel_api_key.get_secret_value(),
            api_token=settings.exotel_api_token.get_secret_value(),
            caller_id=settings.exotel_caller_id,
            app_id=settings.exotel_app_id,
        ),
        asr=primary_asr,
        fallback_asr=fallback_asr,
        asr_by_provider=asr_by_provider,
        llm=llm,
        tts=primary_tts,
        fallback_tts=fallback_tts,
        tts_by_provider=tts_by_provider,
    )
    log.info("Exotel Voicebot applet URL: %s", voicebot_url(settings.public_base_url))
    if settings.inbound_reserved_calls >= settings.max_concurrent_calls:
        log.warning("INBOUND_RESERVED_CALLS reserves every slot: outbound and campaign calls cannot be placed")
    for warning in settings.webhook_configuration_warnings():
        log.warning(warning)
    dispatcher = asyncio.create_task(dispatch.run_forever(app.state.services))
    try:
        yield
    finally:
        dispatcher.cancel()
        with suppress(asyncio.CancelledError):
            await dispatcher
        await http.aclose()
        await pool.close()


def voicebot_url(public_base_url: str) -> str:
    host = public_base_url.split("://", 1)[-1].rstrip("/")
    return f"wss://<EXOTEL_API_KEY>:<EXOTEL_API_TOKEN>@{host}/exotel/stream?sample-rate=16000"


app = FastAPI(title="Voice Agent Platform", lifespan=lifespan)
app.include_router(api.public)
app.include_router(api.admin)
