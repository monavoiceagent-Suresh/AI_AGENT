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
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
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


# Scribe error message types meaning "the connection parameters were rejected by the server" (the SDK's own
# description of `invalid_request`, @elevenlabs/client 1.30.0 scribe/connection.d.ts): the only kind that
# rejected keyterms can cause. Note `unaccepted_terms` is about terms of service, not keyterms.
SCRIBE_REQUEST_REJECTED = frozenset({"invalid_request"})


class ASRError(Exception):
    """A speech-recognition failure. `code` is the provider's own error type (never free text, which may quote
    a caller's words or a persona's terms); `rejected_request` means the provider refused this session's
    configuration (parameters such as keyterms) rather than being unavailable."""

    def __init__(self, message: str, *, code: str = "", rejected_request: bool = False):
        super().__init__(message)
        self.code = code
        self.rejected_request = rejected_request


class ScribeSession:
    def __init__(
        self,
        ws: ClientConnection,
        sample_rate: int,
        *,
        reopen: Callable[[], Awaitable[ClientConnection]] | None = None,
        max_reconnects: int = MAX_RECONNECTS,
        keyterms_sent: int = 0,
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
        self.keyterms_sent = keyterms_sent
        # How many keyterms Scribe's session_started echoed back; None while unconfirmed (never assumed 0).
        self.keyterms_accepted: int | None = None

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
                        self._on_session_started(message)
                    elif "error" in message:
                        rejected = kind in SCRIBE_REQUEST_REJECTED
                        # A rejected request's error text may quote the parameters (a persona's terms): keep it out.
                        detail = "request rejected" if rejected else str(message["error"])[:200]
                        raise ASRError(f"scribe {kind}: {detail}", code=str(kind), rejected_request=rejected)
                cause: BaseException = ConnectionClosedOK(None, None)
            except ConnectionClosed as exc:
                cause = exc
            if self._closed:
                return
            # The server ended the session without being asked to: reopen it rather than end the call.
            await self._reconnect(cause)

    def _on_session_started(self, message: dict) -> None:
        """Scribe echoes the keyterms it is biased towards in session_started.config. Only the counts are logged,
        never the terms (which may be names). A missing echo leaves acceptance unknown, not zero."""
        config = message.get("config")
        echoed = config.get("keyterms") if isinstance(config, dict) else None
        if isinstance(echoed, list):
            self.keyterms_accepted = len(echoed)
        elif isinstance(config, dict) and self.keyterms_sent == 0:
            self.keyterms_accepted = 0  # nothing was sent, so nothing can have been accepted
        accepted = "unknown" if self.keyterms_accepted is None else str(self.keyterms_accepted)
        accepted_count = self.keyterms_accepted
        mismatch = self.keyterms_sent and accepted_count is not None and accepted_count != self.keyterms_sent
        (log.warning if mismatch else log.info)(
            "%s scribe session started asr_event=keyterms_confirmed provider=elevenlabs_scribe "
            "keyterms_sent=%d keyterms_accepted=%s",
            logctx.fields() or "call_id=-",
            self.keyterms_sent,
            accepted,
        )

    async def close(self) -> None:
        self._closed = True
        with suppress(Exception):
            await self._ws.close()


class ScribeRealtime:
    def __init__(
        self, *, api_key: str, model_id: str, silence_ms: int, connect_timeout_ms: int, url: str = REALTIME_URL
    ):
        self._api_key = api_key
        self._url = url
        self._model_id = model_id
        # Scribe accepts 0.192-3.0 s of silence before a VAD commit.
        self._silence_secs = min(3.0, max(0.192, silence_ms / 1000))
        self._connect_timeout = connect_timeout_ms / 1000

    @asynccontextmanager
    async def connect(
        self, *, language: str, sample_rate: int, keyterms: Sequence[str] = ()
    ) -> AsyncIterator[ScribeSession]:
        params: dict[str, object] = {
            "model_id": self._model_id,
            "audio_format": f"pcm_{sample_rate}",
            "language_code": language,
            "commit_strategy": "vad",
            "vad_silence_threshold_secs": self._silence_secs,
        }
        if keyterms:
            # An array query parameter: one `keyterms=` per term. Built per call, so it never leaks between calls.
            params["keyterms"] = list(keyterms)
        query = urlencode(params, doseq=True)

        async def open_socket() -> ClientConnection:
            return await asyncio.wait_for(
                connect(
                    f"{self._url}?{query}",
                    additional_headers={"xi-api-key": self._api_key},
                    open_timeout=self._connect_timeout,
                ),
                self._connect_timeout + 1,
            )

        session = ScribeSession(await open_socket(), sample_rate, reopen=open_socket, keyterms_sent=len(keyterms))
        try:
            yield session
        finally:
            await session.close()
