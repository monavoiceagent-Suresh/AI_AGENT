"""Fetches the real voices available from whichever TTS provider is configured (GET /voices).

Never a hardcoded list: Bakbak and ElevenLabs each publish their own voice catalogue, and what a voice_id
means is specific to the account that owns the API key, so the engine proxies the provider's own answer
instead of guessing at one.
"""

from typing import Any

import httpx

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
        options.append(VoiceOption(voice_id=voice_id, name=name, language=labels.get("language"), model_id=None))
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
