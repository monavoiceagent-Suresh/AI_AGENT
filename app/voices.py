"""Fetches the real voices available from whichever TTS provider is configured (GET /voices).

Never a hardcoded list: Bakbak and ElevenLabs each publish their own voice catalogue, and what a voice_id
means is specific to the account that owns the API key, so the engine proxies the provider's own answer
instead of guessing at one.
"""

import asyncio
import io
import logging
import re
import wave
from contextlib import aclosing
from typing import Any

import httpx

from app.calls import Services, _default_tts_voice
from app.config import Settings, TTSProvider
from app.models import VoiceOption, VoicesOut

log = logging.getLogger(__name__)

BAKBAK_TIMEOUT_SECONDS = 10
ELEVENLABS_VOICES_URL = "https://api.elevenlabs.io/v1/voices"
ELEVENLABS_TIMEOUT_SECONDS = 10
_VOICE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

# Deepgram Aura has no list-voices endpoint: its voices are a fixed catalogue of model names, not an account's
# own data. This is the Aura-2 English catalogue from https://developers.deepgram.com/docs/tts-models
# (fetched 2026-10-31); Aura has no Hindi/Tamil/Telugu voices at all, which is why only English is listed.
DEEPGRAM_AURA_EN_VOICES = (
    "amalthea", "andromeda", "apollo", "arcas", "aries", "asteria", "athena", "atlas", "aurora", "callista",
    "cora", "cordelia", "delia", "draco", "electra", "harmonia", "helena", "hera", "hermes", "hyperion", "iris",
    "janus", "juno", "jupiter", "luna", "mars", "minerva", "neptune", "odysseus", "ophelia", "orion", "orpheus",
    "pandora", "phoebe", "pluto", "saturn", "selene", "thalia", "theia", "vesta", "zeus",
)  # fmt: skip

# Sarvam (Bulbul v4-flash) also has no list-voices endpoint: this is its documented speaker catalogue for the
# engine's four languages, from https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/text-to-speech/
# voices (fetched 2026-11-20). Sarvam has speakers for Marathi/Kannada/Punjabi/Gujarati/Bengali too, but those
# are outside this engine's language set (app/models.py Language) and are not listed here.
SARVAM_V4_FLASH_VOICES: dict[str, tuple[str, ...]] = {
    "en": (
        "aditi_en_stories", "aparna_en_companion", "aparna_en_edtech", "ashwin_en_sports",
        "ashwin_en_sports_energetic", "chandrika_en_stories", "dev_en_recovery", "dev_en_conversational",
        "deven_en_conversation", "ishita_en_customer", "ishita_en_medical", "ishita_en_numbers",
        "ishita_en_social", "ishita_en_stories", "kalpit_en_edtech", "nachiket_en_ads", "neha_en_customer",
        "neha_en_latenight", "nupur_en_kids", "ojas_en_social", "ritu_en_edtech", "ritu_en_latenight",
        "ritu_en_medical", "ritu_en_reels", "rohan_en_recovery", "roopa_en_conversational", "rustom_en_suspense",
        "sanchita_en_companion", "sanchita_en_insurance", "sanchita_en_recovery", "sanchita_en_market",
        "sanchita_en_social", "shabana_en_edtech", "shalini_en_companion", "shalini_en_customer",
        "shubh_en_narration", "shubh_en_numbers", "shubh_en_ads", "shubh_en_recovery", "shubh_en_audiobook",
        "shubh_en_narration_gentle", "shubh_en_sports", "simran_en_narration", "simran_en_automobile",
        "simran_en_conversation", "simran_en_customer", "simran_en_edtech", "simran_en_edtech_bot",
        "simran_en_sales", "simran_en_recovery", "simran_en_ads", "simran_en_therapist", "sunny_en_social",
        "varun_en_ads", "varun_en_suspense", "zarina_en_conversation", "amelia_en_conversational",
        "sophia_en_conversational", "girish_en_documentary", "girish_en_devotional", "payal_en_edtech",
        "sarang_en_narration",
    ),
    "hi": (
        "aayan_hi_conversational", "amit_hi_conversational", "ashutosh_hi_conversational",
        "kabir_hi_conversational", "kavya_hi_conversational", "manan_hi_conversational", "rahul_hi_conversational",
        "sumit_hi_conversational", "aditya_hi_conversational", "aditya_hi_sales", "anand_hi_documentary",
        "anand_hi_news", "aparna_hi_customer", "aparna_hi_kyc", "ashok_hi_character", "ashok_hi_news",
        "chhavi_hi_kids", "ishita_hi_ads", "ishita_hi_edtech", "ishita_hi_banking", "ishita_hi_ads_informal",
        "ishita_hi_devotional", "ishita_hi_numbers", "ishita_hi_social", "kunal_hi_kids",
        "mahesh_hi_documentary", "mani_hi_devotional", "mani_hi_conversational", "mohit_hi_conversational",
        "nachiket_hi_devotional", "priya_hi_recovery", "ratan_hi_latenight", "ratan_hi_customer_expressive",
        "ratan_hi_documentary", "ratan_hi_devotional", "ratan_hi_recovery", "ratan_hi_social",
        "ratan_hi_sports", "ratan_hi_latenight_warm", "rehan_hi_social", "ritu_hi_customer_utility",
        "ritu_hi_kids", "ritu_hi_conversation", "ritu_hi_customer", "ritu_hi_edtech", "ritu_hi_ads_formal",
        "ritu_hi_banking", "ritu_hi_ads_informal", "ritu_hi_insurance", "ritu_hi_edtech_bot", "ritu_hi_medical",
        "ritu_hi_sales", "ritu_hi_reels", "ritu_hi_social", "ritu_hi_customer_warm", "ritu_hi_social_lively",
        "roopa_hi_companion", "roopa_hi_narration", "roopa_hi_recovery", "roopa_hi_market",
        "roopa_hi_conversational", "sanchita_hi_assistant", "sanchita_hi_edtech", "sanchita_hi_banking",
        "sanchita_hi_feedback", "sanchita_hi_ads_formal", "sanchita_hi_ads_informal", "sanchita_hi_interview",
        "sanchita_hi_romantic", "sanchita_hi_market", "sanchita_hi_social", "sanchita_hi_kyc",
        "sarika_hi_conversation", "shalini_hi_companion", "shalini_hi_social", "shreya_hi_conversational",
        "shreya_hi_news", "shruti_hi_edtech", "shubh_hi_customer", "shubh_hi_ecomm", "shubh_hi_stories_mixed",
        "shubh_hi_devotional", "shubh_hi_ads", "shubh_hi_recovery", "shubh_hi_stories_dramatic",
        "simran_hi_assistant", "simran_hi_narration", "simran_hi_automobile", "simran_hi_conversation",
        "simran_hi_news_breaking", "simran_hi_social_energetic", "simran_hi_social_excited",
        "simran_hi_latenight", "simran_hi_news", "simran_hi_recovery", "simran_hi_sales", "suchitra_hi_ecomm",
        "suhani_hi_social", "sunny_hi_ads", "sunny_hi_reels", "tarun_hi_conversational", "tarun_hi_sales",
        "chaitra_hi_customer", "shilpa_hi_narration", "tanya_hi_narration", "aarti_hi_customer",
        "advait_hi_character", "aryaman_hi_ads", "chirag_hi_social", "girish_hi_devotional", "mukul_hi_ads",
        "mukul_hi_suspense", "suman_hi_companion", "vaibhav_hi_social", "vandana_hi_ecomm", "vipul_hi_social",
    ),
    "ta": ("aravind_ta_ads", "aravind_ta_suspense", "gokul_ta_narration", "vijay_ta_narration"),
    "te": ("kavitha_te_conversation", "kavitha_te_narration", "pooja_te_conversation", "tarun_te_narration"),
}


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


def _extra_voice_ids(settings: Settings) -> list[str]:
    ids = [part.strip() for part in settings.elevenlabs_extra_voice_ids.split(",")]
    return [voice_id for voice_id in dict.fromkeys(ids) if _VOICE_ID.match(voice_id)]


async def _with_extra_elevenlabs_voices(
    http: httpx.AsyncClient, settings: Settings, options: list[VoiceOption]
) -> list[VoiceOption]:
    """Append the configured ELEVENLABS_EXTRA_VOICE_IDS that the account's own list lacks. A voice's name and
    language are read from ElevenLabs when it answers; otherwise the voice is still offered under a generic name."""
    known = {option.voice_id for option in options}
    headers = {"xi-api-key": settings.elevenlabs_api_key.get_secret_value()}
    for voice_id in _extra_voice_ids(settings):
        if voice_id in known:
            continue
        found: list[VoiceOption] = []
        try:
            response = await http.get(
                f"{ELEVENLABS_VOICES_URL}/{voice_id}", headers=headers, timeout=ELEVENLABS_TIMEOUT_SECONDS
            )
            if response.status_code < 400:
                found = _parse_elevenlabs([response.json()])
        except (httpx.HTTPError, ValueError):
            pass
        fallback = VoiceOption(voice_id=voice_id, name=f"Custom voice {voice_id[:6]}", language="en")
        options.append(found[0] if found else fallback)
    return options


async def list_voices(http: httpx.AsyncClient, settings: Settings, provider: TTSProvider | None = None) -> VoicesOut:
    """The voices of `provider` (a workspace/persona choice), else of the engine's global TTS_PROVIDER."""
    provider = provider or settings.tts_provider
    if not settings.tts_credentials_present(provider):
        raise VoicesUnavailable(f"TTS provider {provider!r} has no credentials configured on the voice engine")
    if provider == "deepgram_aura":
        options = [
            VoiceOption(voice_id=f"aura-2-{name}-en", name=name.capitalize(), language="en", model_id="aura-2")
            for name in DEEPGRAM_AURA_EN_VOICES
        ]
        return VoicesOut(provider="deepgram", voices=options)
    if provider == "sarvam":
        options = [
            VoiceOption(voice_id=name, name=name.replace("_", " ").title(), language=lang, model_id=settings.sarvam_tts_model)
            for lang, names in SARVAM_V4_FLASH_VOICES.items()
            for name in names
        ]
        return VoicesOut(provider="sarvam", voices=options)
    if provider == "bakbak":
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
        # The provider's own error body never reaches the client (api.py collapses this to a bare 502),
        # so log it here — it's the only place that distinguishes a bad key from a quota/plan limit.
        log.warning("%s voices endpoint failed (%s): %s", vendor, response.status_code, response.text[:500])
        raise VoicesUnavailable(f"{vendor} voices endpoint failed ({response.status_code})")
    try:
        payload = response.json()
    except ValueError as exc:
        raise VoicesUnavailable(f"{vendor} voices endpoint sent an unreadable answer") from exc
    voices = parse(payload)
    if vendor == "elevenlabs":
        voices = await _with_extra_elevenlabs_voices(http, settings, voices)
    return VoicesOut(provider=vendor, voices=voices)
