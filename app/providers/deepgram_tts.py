"""Deepgram Aura text-to-speech over WebSocket.

Reference: https://developers.deepgram.com/reference/text-to-speech-api/speak-streaming
Live-verified (2026-10-31): `Authorization: Token <key>` (not `Bearer`); `encoding=linear16` is raw 16-bit PCM
with no container or header, so unlike Bakbak's float32 SSE frames this needs no conversion; binary frames and
`Metadata`/`Flushed`/`Warning` JSON frames interleave on the same socket.

Aura has no Hindi, Tamil or Telugu voices (confirmed against Deepgram's own model list, and live: sending Tamil
or Hindi text to an English voice does not error, it just mispronounces the text phonetically). This adapter
refuses anything but English rather than risk that silently reaching a caller; `calls.py` is expected to never
select this provider for a non-English agent, but the check here is a second line of defence.
"""

import json
from collections.abc import AsyncIterator
from contextlib import suppress

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from app.interfaces import TTSError

SPEAK_URL = "wss://api.deepgram.com/v1/speak"


class DeepgramTTS:
    def __init__(self, *, api_key: str, url: str = SPEAK_URL):
        self._api_key = api_key
        self._url = url

    async def stream(
        self,
        text: str,
        *,
        voice_id: str,
        model_id: str,
        language: str,
        sample_rate: int,
        previous_text: str,  # Aura has no continuity context; kept for the shared TTS interface
    ) -> AsyncIterator[bytes]:
        if language != "en":
            raise TTSError(f"Deepgram Aura cannot speak language {language!r}; it only has English voices")
        url = f"{self._url}?model={voice_id}&encoding=linear16&sample_rate={sample_rate}"
        try:
            async with connect(url, additional_headers={"Authorization": f"Token {self._api_key}"}) as ws:
                await ws.send(json.dumps({"type": "Speak", "text": text}))
                await ws.send(json.dumps({"type": "Flush"}))
                produced_anything = False
                try:
                    async for raw in ws:
                        if isinstance(raw, bytes):
                            produced_anything = True
                            yield raw
                            continue
                        message = json.loads(raw)
                        kind = message.get("type")
                        if kind == "Flushed":
                            break
                        if kind == "Warning":
                            raise TTSError(f"Deepgram Aura warning: {message.get('description')}")
                finally:
                    with suppress(ConnectionClosed, OSError):
                        await ws.send(json.dumps({"type": "Close"}))
                if not produced_anything:
                    raise TTSError("Deepgram Aura produced no audio")
        except InvalidStatus as exc:
            raise TTSError(f"Deepgram Aura failed ({exc.response.status_code})") from exc
