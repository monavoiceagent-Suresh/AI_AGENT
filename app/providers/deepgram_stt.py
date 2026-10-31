"""Deepgram Nova-3 speech-to-text over WebSocket.

Reference: https://developers.deepgram.com/reference/speech-to-text-api/listen-streaming
Live-verified (2026-10-31) against the real API: `Authorization: Token <key>` (not `Bearer`), audio is sent as
raw binary WebSocket frames (not JSON/base64 like Scribe), `language=ta` is rejected on `nova-2` (HTTP 400) but
accepted on `nova-3`, and `nova-3` transcribes both English and Tamil correctly at 8 kHz and 16 kHz linear16.

Nova-3 is the only model confirmed to cover all four agent languages (en/hi/ta/te), so the model is fixed rather
than configurable per call.
"""

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import urlencode

from websockets.asyncio.client import ClientConnection, connect

from app.interfaces import CommittedTranscript, PartialTranscript
from app.providers.scribe import ASRError

log = logging.getLogger(__name__)

LISTEN_URL = "wss://api.deepgram.com/v1/listen"
MODEL = "nova-3"


class DeepgramSTTSession:
    def __init__(self, ws: ClientConnection):
        self._ws = ws
        # Final-but-not-speech_final segments accumulate here until endpointing closes the utterance.
        self._buffer = ""

    async def send_audio(self, pcm: bytes) -> None:
        await self._ws.send(pcm)

    async def events(self) -> AsyncIterator[PartialTranscript | CommittedTranscript]:
        async for raw in self._ws:
            message = json.loads(raw)
            kind = message.get("type")
            if kind == "Results":
                alternative = message["channel"]["alternatives"][0]
                text = alternative["transcript"]
                if not text:
                    continue
                if not message.get("is_final"):
                    yield PartialTranscript(text)
                elif message.get("speech_final"):
                    committed = f"{self._buffer} {text}".strip() if self._buffer else text
                    self._buffer = ""
                    yield CommittedTranscript(committed)
                else:
                    self._buffer = f"{self._buffer} {text}".strip() if self._buffer else text
                    yield PartialTranscript(self._buffer)
            elif kind == "Error":
                raise ASRError(f"deepgram {message.get('variant', 'error')}: {message.get('description')}")
            # Metadata / SpeechStarted / UtteranceEnd carry nothing this adapter's contract needs.


class DeepgramSTT:
    def __init__(self, *, api_key: str, silence_ms: int, connect_timeout_ms: int, url: str = LISTEN_URL):
        self._api_key = api_key
        # Deepgram's `endpointing` is milliseconds of silence before a segment is marked speech_final.
        self._endpointing_ms = max(10, silence_ms)
        self._connect_timeout = connect_timeout_ms / 1000
        self._url = url

    @asynccontextmanager
    async def connect(self, *, language: str, sample_rate: int) -> AsyncIterator[DeepgramSTTSession]:
        query = urlencode(
            {
                "model": MODEL,
                "language": language,
                "encoding": "linear16",
                "sample_rate": sample_rate,
                "channels": 1,
                "interim_results": "true",
                "punctuate": "true",
                "endpointing": self._endpointing_ms,
            }
        )
        async with connect(
            f"{self._url}?{query}",
            additional_headers={"Authorization": f"Token {self._api_key}"},
            open_timeout=self._connect_timeout,
        ) as ws:
            yield DeepgramSTTSession(ws)
