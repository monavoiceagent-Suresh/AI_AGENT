import base64
import json
from urllib.parse import parse_qs

import httpx
import pytest
from fastapi import WebSocketDisconnect

from app.interfaces import CallAudio, CallEnded, PlaybackMark
from app.providers.exotel import ExotelClient, ExotelError, ExotelMediaStream, chunk_size_for
from tests.fakes import exotel_start

pytestmark = pytest.mark.anyio


class FakeWebSocket:
    def __init__(self, incoming: list[str]) -> None:
        self.incoming = list(incoming)
        self.sent: list[dict] = []

    async def receive_text(self) -> str:
        if not self.incoming:
            raise WebSocketDisconnect(1000)
        return self.incoming.pop(0)

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))


@pytest.mark.parametrize("rate,size", [(8000, 3200), (16000, 3200), (24000, 4800)])
def test_chunk_size_meets_exotel_limits(rate, size):
    assert chunk_size_for(rate) == size
    assert size % 320 == 0 and 3200 <= size <= 100_000


async def test_start_and_inbound_events_are_parsed():
    payload = base64.b64encode(b"\x01\x02" * 800).decode()
    ws = FakeWebSocket(
        [
            json.dumps({"event": "connected"}),
            exotel_start("CA1", sample_rate=16000),
            json.dumps({"event": "media", "stream_sid": "MZstream", "media": {"chunk": "1", "payload": payload}}),
            json.dumps({"event": "dtmf", "dtmf": {"digit": "5", "duration": "100"}}),
            json.dumps({"event": "mark", "mark": {"name": "1:1"}}),
            json.dumps({"event": "stop", "stop": {"call_sid": "CA1", "reason": "callended"}}),
        ]
    )
    media = ExotelMediaStream(ws, queue_max_ms=2000)
    start = await media.wait_start()
    assert (start.call_sid, start.stream_sid, start.sample_rate) == ("CA1", "MZstream", 16000)
    assert start.from_number == "+919876543210"
    events = [e async for e in media.events()]
    assert events == [CallAudio(b"\x01\x02" * 800), PlaybackMark("1:1"), CallEnded("callended")]


async def test_socket_drop_reports_disconnect():
    media = ExotelMediaStream(FakeWebSocket([]), queue_max_ms=2000)
    assert [e async for e in media.events()] == [CallEnded("disconnected")]


async def test_outbound_audio_is_chunked_padded_and_marked():
    ws = FakeWebSocket([exotel_start("CA1", sample_rate=8000)])
    media = ExotelMediaStream(ws, queue_max_ms=60_000)
    await media.wait_start()
    await media.send_audio(bytes(5000))
    await media.flush_audio()
    await media.mark("1:1")
    await media.clear()

    sizes = [len(base64.b64decode(m["media"]["payload"])) for m in ws.sent if m["event"] == "media"]
    assert sizes == [3200, 3200]  # 5000 bytes -> one full chunk + silence-padded tail
    assert all(m["stream_sid"] == "MZstream" for m in ws.sent)
    assert ws.sent[-2] == {"event": "mark", "stream_sid": "MZstream", "mark": {"name": "1:1"}}
    assert ws.sent[-1] == {"event": "clear", "stream_sid": "MZstream"}
    assert media.playback_remaining() == 0.0


async def test_connect_to_flow_request():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"Call": {"Sid": "abc123", "Status": "in-progress"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = ExotelClient(
            http,
            base_url="https://api.exotel.com",
            account_sid="acme1",
            api_key="key",
            api_token="token",
            caller_id="08012345678",
            app_id="99",
        )
        sid = await client.connect_to_flow(
            "+919876543210", status_callback="https://x/cb", time_limit=600, ring_timeout=45
        )

    assert sid == "abc123"
    request = seen[0]
    assert str(request.url) == "https://api.exotel.com/v1/Accounts/acme1/Calls/connect.json"
    assert request.headers["authorization"] == "Basic " + base64.b64encode(b"key:token").decode()
    form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
    assert form == {
        "From": "+919876543210",
        "CallerId": "08012345678",
        "Url": "http://my.exotel.com/acme1/exoml/start_voice/99",
        "TimeLimit": "600",
        "TimeOut": "45",
        "StatusCallback": "https://x/cb",
    }


@pytest.mark.parametrize(
    "answer",
    [
        httpx.ReadTimeout("no answer from Exotel"),
        httpx.ConnectError("unreachable"),
        httpx.Response(401, text="bad credentials"),
        httpx.Response(200, text="<html>not json</html>"),
        httpx.Response(200, json={"Call": {}}),
    ],
)
async def test_connect_to_flow_failures_are_all_exotel_errors(answer):
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(answer, Exception):
            raise answer
        return answer

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = ExotelClient(
            http,
            base_url="https://api.exotel.com",
            account_sid="a",
            api_key="k",
            api_token="t",
            caller_id="0",
            app_id="9",
        )
        with pytest.raises(ExotelError):
            await client.connect_to_flow(
                "+919876543210", status_callback="https://x/cb", time_limit=60, ring_timeout=45
            )
