"""ElevenLabs Scribe Realtime speech-to-text over WebSocket.

Reference: https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime
"""

import base64
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import urlencode

from websockets.asyncio.client import ClientConnection, connect

from app.interfaces import CommittedTranscript, PartialTranscript

log = logging.getLogger(__name__)

REALTIME_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"


class ASRError(Exception):
    pass


class ScribeSession:
    def __init__(self, ws: ClientConnection, sample_rate: int):
        self._ws = ws
        self._sample_rate = sample_rate

    async def send_audio(self, pcm: bytes) -> None:
        await self._ws.send(
            json.dumps(
                {
                    "message_type": "input_audio_chunk",
                    "audio_base_64": base64.b64encode(pcm).decode(),
                    "commit": False,
                    "sample_rate": self._sample_rate,
                }
            )
        )

    async def events(self) -> AsyncIterator[PartialTranscript | CommittedTranscript]:
        async for raw in self._ws:
            message = json.loads(raw)
            kind = message.get("message_type")
            if kind == "partial_transcript":
                yield PartialTranscript(message["text"])
            elif kind == "committed_transcript":
                yield CommittedTranscript(message["text"])
            elif kind == "session_started":
                log.info("scribe session started")
            elif "error" in message:
                raise ASRError(f"scribe {kind}: {message['error']}")


class ScribeRealtime:
    def __init__(self, *, api_key: str, model_id: str, silence_ms: int, connect_timeout_ms: int):
        self._api_key = api_key
        self._model_id = model_id
        # Scribe accepts 0.192-3.0 s of silence before a VAD commit.
        self._silence_secs = min(3.0, max(0.192, silence_ms / 1000))
        self._connect_timeout = connect_timeout_ms / 1000

    @asynccontextmanager
    async def connect(self, *, language: str, sample_rate: int) -> AsyncIterator[ScribeSession]:
        query = urlencode(
            {
                "model_id": self._model_id,
                "audio_format": f"pcm_{sample_rate}",
                "language_code": language,
                "commit_strategy": "vad",
                "vad_silence_threshold_secs": self._silence_secs,
            }
        )
        async with connect(
            f"{REALTIME_URL}?{query}",
            additional_headers={"xi-api-key": self._api_key},
            open_timeout=self._connect_timeout,
        ) as ws:
            yield ScribeSession(ws, sample_rate)
