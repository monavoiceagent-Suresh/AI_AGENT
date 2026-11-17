"""ElevenLabs Scribe Realtime speech-to-text over WebSocket.

Reference: https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime

A Scribe socket that drops mid-call is reopened (bounded attempts) instead of ending the call: the caller's audio
during the gap is buffered (up to a cap) and replayed into the new session, so at most the words spoken in an
over-long gap are lost. A session that cannot be reopened raises ASRError, which ends the call with a recorded
"asr_failed" reason.
"""

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from urllib.parse import urlencode

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

from app import logctx
from app.interfaces import CommittedTranscript, PartialTranscript

log = logging.getLogger(__name__)

REALTIME_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
MAX_RECONNECTS = 2
# Audio kept while reconnecting; older audio is dropped (and counted) rather than growing without bound.
MAX_BUFFERED_SECONDS = 2.0


class ASRError(Exception):
    pass


class ScribeSession:
    def __init__(
        self,
        ws: ClientConnection,
        sample_rate: int,
        *,
        reopen: Callable[[], Awaitable[ClientConnection]] | None = None,
        max_reconnects: int = MAX_RECONNECTS,
    ):
        self._ws = ws
        self._sample_rate = sample_rate
        self._reopen = reopen
        self._max_reconnects = max_reconnects
        self._reconnects = 0
        self._reconnecting = False
        self._buffer = bytearray()
        self._dropped_bytes = 0
        self._closed = False

    @property
    def reconnects(self) -> int:
        return self._reconnects

    def _frame(self, pcm: bytes) -> str:
        return json.dumps(
            {
                "message_type": "input_audio_chunk",
                "audio_base_64": base64.b64encode(pcm).decode(),
                "commit": False,
                "sample_rate": self._sample_rate,
            }
        )

    def _hold(self, pcm: bytes) -> None:
        self._buffer.extend(pcm)
        cap = int(MAX_BUFFERED_SECONDS * self._sample_rate * 2)
        if len(self._buffer) > cap:
            excess = len(self._buffer) - cap
            del self._buffer[:excess]
            self._dropped_bytes += excess

    async def send_audio(self, pcm: bytes) -> None:
        if self._reconnecting:
            self._hold(pcm)
            return
        try:
            await self._ws.send(self._frame(pcm))
        except ConnectionClosed:
            if self._reopen is None:
                raise
            # events() notices the same closure and reopens; keep this audio for the new session.
            self._hold(pcm)

    async def _reconnect(self, cause: BaseException) -> None:
        if self._reopen is None or self._reconnects >= self._max_reconnects:
            raise ASRError(f"scribe disconnected ({type(cause).__name__}) and could not be reopened") from cause
        self._reconnects += 1
        self._reconnecting = True
        log.warning(
            "%s scribe disconnected (%s); reconnecting (attempt %d of %d)",
            logctx.fields() or "call_id=-",
            type(cause).__name__,
            self._reconnects,
            self._max_reconnects,
        )
        try:
            self._ws = await self._reopen()
        except Exception as exc:
            raise ASRError(f"scribe reconnect failed ({type(exc).__name__})") from exc
        finally:
            self._reconnecting = False
        held, self._buffer = bytes(self._buffer), bytearray()
        if held:
            await self._ws.send(self._frame(held))
        log.info(
            "%s scribe reconnected; replayed_ms=%d dropped_ms=%d",
            logctx.fields() or "call_id=-",
            round(len(held) / (2 * self._sample_rate) * 1000),
            round(self._dropped_bytes / (2 * self._sample_rate) * 1000),
        )
        self._dropped_bytes = 0

    async def events(self) -> AsyncIterator[PartialTranscript | CommittedTranscript]:
        while True:
            try:
                async for raw in self._ws:
                    message = json.loads(raw)
                    kind = message.get("message_type")
                    if kind == "partial_transcript":
                        yield PartialTranscript(message["text"])
                    elif kind == "committed_transcript":
                        yield CommittedTranscript(message["text"])
                    elif kind == "session_started":
                        log.info("%s scribe session started", logctx.fields() or "call_id=-")
                    elif "error" in message:
                        raise ASRError(f"scribe {kind}: {message['error']}")
                cause: BaseException = ConnectionClosedOK(None, None)
            except ConnectionClosed as exc:
                cause = exc
            if self._closed:
                return
            # The server ended the session without being asked to: reopen it rather than end the call.
            await self._reconnect(cause)

    async def close(self) -> None:
        self._closed = True
        with suppress(Exception):
            await self._ws.close()


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

        async def open_socket() -> ClientConnection:
            return await asyncio.wait_for(
                connect(
                    f"{REALTIME_URL}?{query}",
                    additional_headers={"xi-api-key": self._api_key},
                    open_timeout=self._connect_timeout,
                ),
                self._connect_timeout + 1,
            )

        session = ScribeSession(await open_socket(), sample_rate, reopen=open_socket)
        try:
            yield session
        finally:
            await session.close()
