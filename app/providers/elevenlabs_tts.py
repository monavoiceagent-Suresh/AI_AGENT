"""ElevenLabs streaming text-to-speech, returning raw PCM for the telephony leg.

Reference: https://elevenlabs.io/docs/api-reference/text-to-speech/stream

Every failure surfaces as a TTSError naming its category (HTTP status, timeout, connection), never as a bare
transport exception, so the session can log and record exactly why a sentence produced no audio. A successful
response here only means audio was synthesized: whether it was sent and played is tracked by the session.
"""

from collections.abc import AsyncIterator

import httpx

from app.interfaces import TTSError

# Connect quickly or give up (the session then tries its fallback TTS before any audio); between audio chunks
# allow longer, since a long sentence legitimately streams for a while.
CONNECT_TIMEOUT_SECONDS = 3.0


class ElevenLabsTTS:
    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        api_key: str,
        stability: float,
        similarity_boost: float,
        read_timeout_s: float = 5.0,
    ):
        self._http = http
        self._headers = {"xi-api-key": api_key}
        self._voice_settings = {"stability": stability, "similarity_boost": similarity_boost}
        self._timeout = httpx.Timeout(read_timeout_s, connect=CONNECT_TIMEOUT_SECONDS)

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
            "voice_settings": {
                **self._voice_settings,
                **{k: v for k, v in (voice_settings or {}).items() if v is not None},
            },
        }
        # eleven_v3 (no longer selectable, kept defensive) rejects text continuity fields, even previous_text=null.
        if model_id != "eleven_v3":
            body["previous_text"] = previous_text or None
        # multilingual_v2 rejects language_code; the other supported models accept it.
        if model_id != "eleven_multilingual_v2":
            body["language_code"] = language
        try:
            async with self._http.stream(
                "POST",
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream",
                params={"output_format": f"pcm_{sample_rate}"},
                headers=self._headers,
                json=body,
                timeout=self._timeout,
            ) as response:
                if response.status_code >= 400:
                    detail = (await response.aread())[:300]
                    raise TTSError(f"ElevenLabs TTS failed (HTTP {response.status_code}, model {model_id}): {detail!r}")
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk
        except httpx.TimeoutException as exc:
            raise TTSError(f"ElevenLabs TTS timed out ({type(exc).__name__}, model {model_id})") from exc
        except httpx.TransportError as exc:
            raise TTSError(f"ElevenLabs TTS connection failed ({type(exc).__name__}, model {model_id})") from exc
