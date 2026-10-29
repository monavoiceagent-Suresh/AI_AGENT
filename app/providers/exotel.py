"""Exotel telephony: Connect-to-flow REST API and the Voicebot applet WebSocket protocol.

Protocol reference: https://developer.exotel.com/docs/agentstream/websocket-protocol
"""

import asyncio
import base64
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import WebSocket, WebSocketDisconnect

from app.interfaces import CallAudio, CallEnded, CallEvent, PlaybackMark
from app.redaction import redact

log = logging.getLogger(__name__)

# Outbound media chunks must be 3200..100000 bytes and a multiple of 320 bytes.
MIN_CHUNK_BYTES = 3200
CHUNK_MULTIPLE = 320
# The shared HTTP client's default timeout is the TTS one (seconds); placing a call may take longer.
CONNECT_TIMEOUT_SECONDS = 15


class ExotelError(Exception):
    pass


@dataclass(frozen=True)
class ExotelCallRecord:
    """What Exotel's own call record says about a call (GET /v1/Accounts/{sid}/Calls/{CallSid}.json)."""

    to: str
    phone_number_sid: str
    direction: str
    forwarded_from: str


class ExotelClient:
    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        base_url: str,
        account_sid: str,
        api_key: str,
        api_token: str,
        caller_id: str,
        app_id: str,
    ):
        self._http = http
        self._base_url = base_url.rstrip("/")
        self._account_sid = account_sid
        self._auth = (api_key, api_token)
        self._caller_id = caller_id
        self._app_id = app_id

    async def get_call_record(self, call_sid: str, *, wait_seconds: float) -> ExotelCallRecord:
        """Exotel's record of a call, read with the account's own credentials. Read-only."""
        response = await self._http.get(
            f"{self._base_url}/v1/Accounts/{self._account_sid}/Calls/{call_sid}.json",
            auth=self._auth,
            timeout=wait_seconds,
        )
        if response.status_code != 200:
            raise ExotelError(f"Exotel call lookup failed ({response.status_code})")
        call = response.json().get("Call") or {}
        return ExotelCallRecord(
            to=str(call.get("To") or ""),
            phone_number_sid=str(call.get("PhoneNumberSid") or ""),
            direction=str(call.get("Direction") or ""),
            forwarded_from=str(call.get("ForwardedFrom") or ""),
        )

    async def connect_to_flow(self, number: str, *, status_callback: str, time_limit: int, ring_timeout: int) -> str:
        """Dial `number` and, once answered, run the App Bazaar flow. Returns the Exotel CallSid."""
        if not self._app_id:
            raise ExotelError("EXOTEL_APP_ID is not configured")
        try:
            response = await self._http.post(
                f"{self._base_url}/v1/Accounts/{self._account_sid}/Calls/connect.json",
                auth=self._auth,
                data={
                    "From": number,
                    "CallerId": self._caller_id,
                    "Url": f"http://my.exotel.com/{self._account_sid}/exoml/start_voice/{self._app_id}",
                    "TimeLimit": str(time_limit),
                    "TimeOut": str(ring_timeout),
                    # Terminal-state callbacks (CallSid, Status) are sent by default.
                    "StatusCallback": status_callback,
                },
                timeout=CONNECT_TIMEOUT_SECONDS,
            )
            if response.status_code >= 400:
                raise ExotelError(f"Exotel connect failed ({response.status_code}): {response.text[:500]}")
            return response.json()["Call"]["Sid"]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            # A timeout or a bad body: the call may or may not have been placed, but it is never an unhandled 500
            # on POST /calls, and the dispatcher settles the attempt instead of leaving its lead claimed.
            raise ExotelError(f"Exotel connect failed: {type(exc).__name__}") from exc


@dataclass(frozen=True)
class StreamStart:
    stream_sid: str
    call_sid: str
    from_number: str
    to_number: str
    sample_rate: int
    custom_parameters: dict[str, str] = field(default_factory=dict)
    # The number a call was routed by, when it differs from `to_number` (set once a route is found).
    called_number: str | None = None
    # Redacted view of the frames received before the media starts, for logs only.
    diagnostics: dict[str, Any] = field(default_factory=dict, compare=False)


def chunk_size_for(sample_rate: int) -> int:
    """~100 ms of 16-bit audio, honouring Exotel's minimum and 320-byte multiple."""
    size = sample_rate * 2 // 10
    size -= size % CHUNK_MULTIPLE
    return max(MIN_CHUNK_BYTES, size)


class ExotelMediaStream:
    """Bidirectional media over the WebSocket opened by Exotel's Voicebot applet."""

    def __init__(self, ws: WebSocket, *, queue_max_ms: int):
        self._ws = ws
        self._queue_max = queue_max_ms / 1000
        self.stream_sid = ""
        self.sample_rate = 8000
        self._chunk_size = chunk_size_for(self.sample_rate)
        self._buffer = bytearray()
        self._play_until = 0.0  # monotonic time at which sent audio finishes playing

    async def wait_start(self) -> StreamStart:
        before_start: list[dict[str, Any]] = []
        while True:
            message = json.loads(await self._ws.receive_text())
            if message.get("event") != "start":
                before_start.append({"event": message.get("event"), "keys": sorted(message)})
                continue
            start = message["start"]
            diagnostics = {
                "events_before_start": before_start,
                "message_keys": sorted(message),
                "start_keys": sorted(start),
                # The business-side identifiers are shown in full; the caller and everything else is masked.
                "start": redact(start, keep_full=frozenset({"to", "call_sid", "stream_sid"})),
            }
            log.info("exotel stream start %s", json.dumps(diagnostics, sort_keys=True, default=str))
            self.stream_sid = start["stream_sid"]
            self.sample_rate = int(start.get("media_format", {}).get("sample_rate", 8000))
            self._chunk_size = chunk_size_for(self.sample_rate)
            return StreamStart(
                stream_sid=self.stream_sid,
                call_sid=start["call_sid"],
                from_number=start.get("from", ""),
                to_number=start.get("to", ""),
                sample_rate=self.sample_rate,
                custom_parameters=start.get("custom_parameters") or {},
                diagnostics=diagnostics,
            )

    async def events(self) -> AsyncIterator[CallEvent]:
        try:
            while True:
                message = json.loads(await self._ws.receive_text())
                event = message.get("event")
                if event == "media":
                    yield CallAudio(base64.b64decode(message["media"]["payload"]))
                elif event == "mark":
                    yield PlaybackMark(message["mark"]["name"])
                elif event == "stop":
                    yield CallEnded(message.get("stop", {}).get("reason", "stopped"))
                    return
                elif event == "dtmf":
                    log.info("dtmf digit received (ignored)")
        except WebSocketDisconnect:
            yield CallEnded("disconnected")

    async def send_audio(self, pcm: bytes) -> None:
        self._buffer.extend(pcm)
        while len(self._buffer) >= self._chunk_size:
            chunk = bytes(self._buffer[: self._chunk_size])
            del self._buffer[: self._chunk_size]
            await self._send_chunk(chunk)

    async def flush_audio(self) -> None:
        if self._buffer:
            # Pad the tail with silence to a valid chunk size.
            chunk = bytes(self._buffer) + bytes(self._chunk_size - len(self._buffer))
            self._buffer.clear()
            await self._send_chunk(chunk)

    async def _send_chunk(self, chunk: bytes) -> None:
        # Stay at most queue_max ahead of playback so interruptions discard little audio.
        ahead = self._play_until - time.monotonic()
        if ahead > self._queue_max:
            await asyncio.sleep(ahead - self._queue_max)
        now = time.monotonic()
        self._play_until = max(self._play_until, now) + len(chunk) / (2 * self.sample_rate)
        await self._send(
            {"event": "media", "stream_sid": self.stream_sid, "media": {"payload": base64.b64encode(chunk).decode()}}
        )

    async def mark(self, name: str) -> None:
        await self._send({"event": "mark", "stream_sid": self.stream_sid, "mark": {"name": name}})

    async def clear(self) -> None:
        self._buffer.clear()
        self._play_until = 0.0
        await self._send({"event": "clear", "stream_sid": self.stream_sid})

    def playback_remaining(self) -> float:
        return max(0.0, self._play_until - time.monotonic())

    async def close(self) -> None:
        """Closing the socket hands the call to the next applet in the Exotel flow."""
        with suppress(RuntimeError, WebSocketDisconnect, OSError):
            await self._ws.close()

    async def _send(self, message: dict) -> None:
        try:
            await self._ws.send_text(json.dumps(message))
        except (WebSocketDisconnect, RuntimeError, OSError):
            # Caller hung up mid-send; events() reports the disconnect.
            pass
