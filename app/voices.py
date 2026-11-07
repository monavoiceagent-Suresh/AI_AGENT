"""Fetches the real voices available from whichever TTS provider is configured (GET /voices).

Never a hardcoded list: Bakbak and ElevenLabs each publish their own voice catalogue, and what a voice_id
means is specific to the account that owns the API key, so the engine proxies the provider's own answer
instead of guessing at one.
"""

import asyncio
import io
import wave
from contextlib import aclosing
from typing import Any

import httpx

from app.calls import Services, _default_tts_voice
from app.config import Settings
from app.models import VoiceOption, VoicesOut

BAKBAK_TIMEOUT_SECONDS = 10
ELEVENLABS_VOICES_URL = "https://api.elevenlabs.io/v1/voices"
ELEVENLABS_TIMEOUT_SECONDS = 10

# Deepgram Aura has no list-voices endpoint: its voices are a fixed catalogue of model names, not an account's
# own data. This is the Aura-2 English catalogue from https://developers.deepgram.com/docs/tts-models
# (fetched 2026-10-31); Aura has no Hindi/Tamil/Telugu voices at all, which is why only English is listed.
DEEPGRAM_AURA_EN_VOICES = (
    "amalthea", "andromeda", "apollo", "arcas", "aries", "asteria", "athena", "atlas", "aurora", "callista",
    "cora", "cordelia", "delia", "draco", "electra", "harmonia", "helena", "hera", "hermes", "hyperion", "iris",
    "janus", "juno", "jupiter", "luna", "mars", "minerva", "neptune", "odysseus", "ophelia", "orion", "orpheus",
    "pandora", "phoebe", "pluto", "saturn", "selene", "thalia", "theia", "vesta", "zeus",
)  # fmt: skip


class VoicesUnavailable(Exception):
    """The provider's voices endpoint could not be reached or answered with an error."""


_LANGUAGE_NAMES = {"english": "en", "hindi": "hi", "tamil": "ta", "telugu": "te"}


def _normalize_language(label: str | None) -> str:
    """ElevenLabs' own language codes/names, folded into the engine's "en"/"hi"/"ta"/"te". Untagged voices
    (most of ElevenLabs' default catalogue) default to "en" — they are English, just not labelled as such.
    A genuinely unrecognized value is kept as-is rather than discarded."""
    key = (label or "en").lower().replace("_", "-").split("-", 1)[0]
    return _LANGUAGE_NAMES.get(key, key)


_PREVIEW_TEXT = {
    "en": "Hello! This is a preview of how my voice sounds on Mona.",
    "hi": "नमस्ते! यह मोना पर मेरी आवाज़ का एक नमूना है।",
    "ta": "வணக்கம்! இது மோனாவில் என் குரலின் ஒரு மாதிரி.",
    "te": "నమస్కారం! ఇది మోనాలో నా స్వరానికి ఒక నమూనా.",
}


async def preview_voice(svc: Services, voice_id: str) -> bytes | None:
    """Render a fixed sample with a voice from the same catalogue used by the picker."""
    catalogue = await list_voices(svc.http, svc.settings)
    voice = next((option for option in catalogue.voices if option.voice_id == voice_id), None)
    if voice is None:
        return None
    language = _normalize_language(voice.language)
    if language not in _PREVIEW_TEXT:
        language = "en"
    default = _default_tts_voice(svc.settings.tts_provider, svc.settings, language)
    if default is None and svc.settings.tts_provider != "bakbak":
        language = "en"
        default = _default_tts_voice(svc.settings.tts_provider, svc.settings, language)
    model = voice.model_id or (default.model_id if default else None)
    if not model:
        raise VoicesUnavailable("Preview model unavailable")
    sample_rate = 16000
    pcm = bytearray()
    stream = svc.tts.stream(_PREVIEW_TEXT[language], voice_id=voice.voice_id, model_id=model,
                            language=language, sample_rate=sample_rate, previous_text="")
    async with asyncio.timeout(15), aclosing(stream):
        async for chunk in stream:
            pcm.extend(chunk[: sample_rate * 20 - len(pcm)])
            if len(pcm) >= sample_rate * 20:
                break
    if len(pcm) < 2:
        raise VoicesUnavailable("Provider returned no audio")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
        wav.writeframes(pcm[:len(pcm) - len(pcm) % 2])
    return output.getvalue()


def _as_list(payload: Any) -> list[dict[str, Any]]:
    """Bakbak/ElevenLabs may wrap their list under a key; accept a bare list too rather than guess which."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("voices", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _text(item: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _parse_bakbak(payload: Any) -> list[VoiceOption]:
    options = []
    for item in _as_list(payload):
        voice_id, name = _text(item, "voice_id", "id"), _text(item, "name", "voice_name")
        if not voice_id or not name:
            continue  # a malformed entry is skipped, not guessed at
        options.append(
            VoiceOption(
                voice_id=voice_id,
                name=name,
                language=_text(item, "language", "language_code"),
                model_id=_text(item, "model_id", "model"),
            )
        )
    return options


def _parse_elevenlabs(payload: Any) -> list[VoiceOption]:
    options = []
    for item in _as_list(payload):
        voice_id, name = _text(item, "voice_id"), _text(item, "name")
        if not voice_id or not name:
            continue
        labels = item.get("labels") if isinstance(item.get("labels"), dict) else {}
        options.append(
            VoiceOption(voice_id=voice_id, name=name, language=_normalize_language(labels.get("language")), model_id=None)
        )
    return options


async def list_voices(http: httpx.AsyncClient, settings: Settings) -> VoicesOut:
    if settings.tts_provider == "deepgram_aura":
        options = [
            VoiceOption(voice_id=f"aura-2-{name}-en", name=name.capitalize(), language="en", model_id="aura-2")
            for name in DEEPGRAM_AURA_EN_VOICES
        ]
        return VoicesOut(provider="deepgram", voices=options)
    if settings.tts_provider == "bakbak":
        vendor, url = "bakbak", f"{settings.bakbak_base_url.rstrip('/')}/voices"
        headers = {"X-API-Key": settings.bakbak_api_key.get_secret_value()}
        timeout = BAKBAK_TIMEOUT_SECONDS
        parse = _parse_bakbak
    else:
        vendor = "elevenlabs"
        url, headers, timeout = (
            ELEVENLABS_VOICES_URL,
            {"xi-api-key": settings.elevenlabs_api_key.get_secret_value()},
            ELEVENLABS_TIMEOUT_SECONDS,
        )
        parse = _parse_elevenlabs
    try:
        response = await http.get(url, headers=headers, timeout=timeout)
    except httpx.HTTPError as exc:
        raise VoicesUnavailable(f"{vendor} voices endpoint unreachable: {type(exc).__name__}") from exc
    if response.status_code >= 400:
        raise VoicesUnavailable(f"{vendor} voices endpoint failed ({response.status_code})")
    try:
        payload = response.json()
    except ValueError as exc:
        raise VoicesUnavailable(f"{vendor} voices endpoint sent an unreadable answer") from exc
    return VoicesOut(provider=vendor, voices=parse(payload))
