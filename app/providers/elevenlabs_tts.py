"""ElevenLabs streaming text-to-speech, returning raw PCM for the telephony leg.

Reference: https://elevenlabs.io/docs/api-reference/text-to-speech/stream
"""

from collections.abc import AsyncIterator

import httpx

from app.interfaces import TTSError


class ElevenLabsTTS:
    def __init__(self, http: httpx.AsyncClient, *, api_key: str, stability: float, similarity_boost: float):
        self._http = http
        self._headers = {"xi-api-key": api_key}
        self._voice_settings = {"stability": stability, "similarity_boost": similarity_boost}

    async def stream(
        self,
        text: str,
        *,
        voice_id: str,
        model_id: str,
        language: str,
        sample_rate: int,
        previous_text: str,
        voice_settings: dict | None = None,
    ) -> AsyncIterator[bytes]:
        body = {
            "text": text,
            "model_id": model_id,
            # The call's own override (e.g. a campaign's chosen tone) wins per-field; anything it
            # doesn't set keeps this provider instance's fixed default.
            "voice_settings": {**self._voice_settings, **{k: v for k, v in (voice_settings or {}).items() if v is not None}},
        }
        # eleven_v3 rejects text continuity fields, including previous_text set to null.
        if model_id != "eleven_v3":
            body["previous_text"] = previous_text or None
        # multilingual_v2 rejects language_code; the other supported models accept it.
        if model_id != "eleven_multilingual_v2":
            body["language_code"] = language
        async with self._http.stream(
            "POST",
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream",
            params={"output_format": f"pcm_{sample_rate}"},
            headers=self._headers,
            json=body,
        ) as response:
            if response.status_code >= 400:
                detail = (await response.aread())[:500]
                raise TTSError(f"ElevenLabs TTS failed ({response.status_code}): {detail!r}")
            async for chunk in response.aiter_bytes():
                yield chunk
