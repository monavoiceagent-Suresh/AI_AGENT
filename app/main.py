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
from app.providers.elevenlabs_tts import ElevenLabsTTS
from app.providers.exotel import ExotelClient
from app.providers.failover import FailoverLLM
from app.providers.scribe import ScribeRealtime

log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pool = await db.create_pool(settings.database_url, settings.database_pool_min_size, settings.database_pool_max_size)
    if applied := await db.migrate(pool):
        log.info("applied migrations: %s", ", ".join(applied))
    # One pooled client keeps TLS connections warm to Exotel, ElevenLabs, Groq and OpenRouter.
    http = httpx.AsyncClient(timeout=settings.tts_timeout_ms / 1000)
    elevenlabs_tts = ElevenLabsTTS(
        http,
        api_key=settings.elevenlabs_api_key.get_secret_value(),
        stability=settings.elevenlabs_tts_stability,
        similarity_boost=settings.elevenlabs_tts_similarity_boost,
    )
    if settings.tts_provider == "bakbak":
        primary_tts = BakbakTTS(
            http, api_key=settings.bakbak_api_key.get_secret_value(), base_url=settings.bakbak_base_url
        )
        fallback_tts = elevenlabs_tts if settings.tts_fallback_enabled else None
    else:
        primary_tts, fallback_tts = elevenlabs_tts, None
    primary = ChatCompletionsLLM(
        http,
        name="OpenRouter",
        api_key=settings.openrouter_api_key.get_secret_value(),
        base_url=settings.openrouter_base_url,
        max_tokens=settings.openrouter_max_output_tokens,
        temperature=settings.openrouter_temperature,
        timeout_ms=settings.openrouter_timeout_ms,
        max_retries=settings.openrouter_max_retries,
    )
    fallback: ChatCompletionsLLM | None = None
    if settings.groq_api_key:
        fallback = ChatCompletionsLLM(
            http,
            name="Groq",
            api_key=settings.groq_api_key.get_secret_value(),
            base_url=settings.groq_base_url,
            max_tokens=settings.groq_max_output_tokens,
            temperature=settings.groq_temperature,
            timeout_ms=settings.groq_timeout_ms,
            max_retries=settings.groq_max_retries,
        )
    llm = FailoverLLM(primary=primary, fallback=fallback, fallback_model=settings.groq_model)
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
        asr=ScribeRealtime(
            api_key=settings.elevenlabs_api_key.get_secret_value(),
            model_id=settings.elevenlabs_asr_model_id,
            silence_ms=settings.endpointing_min_silence_ms,
            connect_timeout_ms=settings.asr_timeout_ms,
        ),
        llm=llm,
        tts=primary_tts,
        fallback_tts=fallback_tts,
    )
    log.info("Exotel Voicebot applet URL: %s", voicebot_url(settings.public_base_url))
    if settings.inbound_reserved_calls >= settings.max_concurrent_calls:
        log.warning("INBOUND_RESERVED_CALLS reserves every slot: outbound and campaign calls cannot be placed")
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
