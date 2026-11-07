"""Bakbak (Raya) streaming text-to-speech, returning 16-bit PCM for the telephony leg.

Reference: https://docs.litwizlabs.com/api-reference/text-to-speech/text-to-speech-stream
The endpoint answers with Server-Sent Events whose `chunk` events carry base64 PCM float32 little-endian audio.
"""

import base64
import json
import struct
from collections.abc import AsyncIterator

import httpx

from app.interfaces import TTSError

# Our agent languages -> Bakbak `language` codes (Bakbak names Indian English "en-in").
LANGUAGE_CODES = {"en": "en-in", "hi": "hi", "ta": "ta", "te": "te"}


def float32_to_pcm16(data: bytes) -> tuple[bytes, bytes]:
    """Convert whole float32 samples to clipped int16; returns (pcm16, trailing partial sample)."""
    usable = len(data) - len(data) % 4
    count = usable // 4
    samples = struct.unpack(f"<{count}f", data[:usable])
    pcm = struct.pack(f"<{count}h", *(max(-32768, min(32767, round(s * 32767))) for s in samples))
    return pcm, data[usable:]


class BakbakTTS:
    def __init__(self, http: httpx.AsyncClient, *, api_key: str, base_url: str):
        self._http = http
        self._url = f"{base_url.rstrip('/')}/text-to-speech/stream"
        self._headers = {"X-API-Key": api_key, "Accept": "text/event-stream"}

    async def stream(
        self,
        text: str,
        *,
        voice_id: str,
        model_id: str,
        language: str,
        sample_rate: int,
        previous_text: str,  # Bakbak has no continuity context; kept for the shared TTS interface
        voice_settings: dict | None = None,  # Bakbak has no tone controls; kept for the shared TTS interface
    ) -> AsyncIterator[bytes]:
        body = {
            "text": text,
            "voice_id": voice_id,
            "model": model_id,
            "language": LANGUAGE_CODES[language],
            "codec": "pcm",
            "sample_rate": sample_rate,
        }
        async with self._http.stream("POST", self._url, headers=self._headers, json=body) as response:
            if response.status_code >= 400:
                detail = (await response.aread())[:500]
                raise TTSError(f"Bakbak TTS failed ({response.status_code}): {detail!r}")
            partial = b""
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[len("data:") :])
                if event.get("type") == "chunk":
                    pcm, partial = float32_to_pcm16(partial + base64.b64decode(event["data"]))
                    if pcm:
                        yield pcm
                elif event.get("type") == "done":
                    return
                elif event.get("type") == "error" or "error" in event:
                    raise TTSError(f"Bakbak TTS stream error: {event}")
            raise TTSError("Bakbak TTS stream ended without a done event")
