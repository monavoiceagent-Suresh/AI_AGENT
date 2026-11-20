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
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from urllib.parse import urlencode

from websockets.asyncio.client import ClientConnection, connect

from app.interfaces import CommittedTranscript, PartialTranscript
from app.providers.scribe import ASRError

log = logging.getLogger(__name__)

LISTEN_URL = "wss://api.deepgram.com/v1/listen"
MODEL = "nova-3"


def _deepgram_error(message: dict, keyterms_sent: int) -> ASRError:
    """Deepgram's streaming Error message as an ASRError. Deepgram documents no error code for a rejected
    keyterm after the handshake (oversized keyterms are refused at the handshake with HTTP 400), so an Error is
    treated as a rejected request only when it names keyterms and this session sent some. Unverified live."""
    variant = str(message.get("variant") or "error")
    text = f"{message.get('description') or ''} {message.get('message') or ''}"
    rejected = bool(keyterms_sent) and "keyterm" in text.lower()
    detail = "request rejected" if rejected else text.strip()[:200]
    return ASRError(f"deepgram {variant}: {detail}", code=variant, rejected_request=rejected)


class DeepgramSTTSession:
    def __init__(self, ws: ClientConnection, *, keyterms_sent: int = 0):
        self._ws = ws
        self.keyterms_sent = keyterms_sent
        # Deepgram never echoes the keyterms it applied: acceptance is always unknown.
        self.keyterms_accepted: int | None = None
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
                raise _deepgram_error(message, self.keyterms_sent)
            # Metadata / SpeechStarted / UtteranceEnd carry nothing this adapter's contract needs.


class DeepgramSTT:
    def __init__(self, *, api_key: str, silence_ms: int, connect_timeout_ms: int, url: str = LISTEN_URL):
        self._api_key = api_key
        # Deepgram's `endpointing` is milliseconds of silence before a segment is marked speech_final.
        self._endpointing_ms = max(10, silence_ms)
        self._connect_timeout = connect_timeout_ms / 1000
        self._url = url

    @asynccontextmanager
    async def connect(
        self, *, language: str, sample_rate: int, keyterms: Sequence[str] = ()
    ) -> AsyncIterator[DeepgramSTTSession]:
        params: dict[str, object] = {
            "model": MODEL,
            "language": language,
            "encoding": "linear16",
            "sample_rate": sample_rate,
            "channels": 1,
            "interim_results": "true",
            "punctuate": "true",
            "endpointing": self._endpointing_ms,
        }
        if keyterms:
            # Nova-3 keyterm prompting: the parameter is repeated, one term each (not comma-separated).
            params["keyterm"] = list(keyterms)
        query = urlencode(params, doseq=True)
        async with connect(
            f"{self._url}?{query}",
            additional_headers={"Authorization": f"Token {self._api_key}"},
            open_timeout=self._connect_timeout,
        ) as ws:
            yield DeepgramSTTSession(ws, keyterms_sent=len(keyterms))
