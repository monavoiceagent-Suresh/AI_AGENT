"""Deepgram Nova-3 ASR adapter: protocol mapping against a local fake of the real WebSocket API."""

import asyncio
import json
from contextlib import suppress
from urllib.parse import parse_qs, urlsplit

import pytest
from websockets.asyncio.server import serve

from app.interfaces import CommittedTranscript, PartialTranscript
from app.providers.deepgram_stt import DeepgramSTT
from app.providers.scribe import ASRError

pytestmark = pytest.mark.anyio

KEY = "deepgram-test-key"


def results(transcript: str, *, is_final: bool, speech_final: bool = False) -> dict:
    return {
        "type": "Results",
        "is_final": is_final,
        "speech_final": speech_final,
        "channel": {"alternatives": [{"transcript": transcript}]},
    }


class FakeDeepgramListen:
    """Replays a scripted sequence of server messages, in response to the client's audio frames."""

    def __init__(self, messages: list[dict], *, expect_audio_frames: int = 0):
        self.messages = messages
        self.expect_audio_frames = expect_audio_frames
        self.received_audio: list[bytes] = []
        self.auth_header: str | None = None
        self.query: dict[str, list[str]] = {}

    async def handler(self, ws):
        self.auth_header = ws.request.headers.get("Authorization")
        self.query = parse_qs(urlsplit(ws.request.path).query)

        async def drain_audio() -> None:
            # The real API is bidirectional: audio keeps streaming in while results stream out, so draining
            # must not block sending the scripted messages below (sequential send-then-receive deadlocks).
            async for raw in ws:
                if isinstance(raw, bytes):
                    self.received_audio.append(raw)

        drain_task = asyncio.ensure_future(drain_audio())
        # Wait for the audio the test expects (not a fixed sleep: that races under load), capped so a
        # genuine protocol bug still fails fast instead of hanging.
        for _ in range(200):
            if len(self.received_audio) >= self.expect_audio_frames:
                break
            await asyncio.sleep(0.01)
        for message in self.messages:
            await ws.send(json.dumps(message))
        drain_task.cancel()
        with suppress(asyncio.CancelledError):
            await drain_task


async def run(fake: FakeDeepgramListen, audio_chunks: list[bytes], *, language="ta", sample_rate=16000):
    async with serve(fake.handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        stt = DeepgramSTT(api_key=KEY, silence_ms=300, connect_timeout_ms=2000, url=f"ws://127.0.0.1:{port}/v1/listen")
        events = []
        async with stt.connect(language=language, sample_rate=sample_rate) as session:
            for chunk in audio_chunks:
                await session.send_audio(chunk)
            async for event in session.events():
                events.append(event)
    return events


async def test_sends_raw_binary_audio_with_token_auth_and_the_right_query():
    fake = FakeDeepgramListen([results("வணக்கம்", is_final=True, speech_final=True)], expect_audio_frames=2)
    events = await run(fake, [b"\x01\x02", b"\x03\x04"], language="ta", sample_rate=8000)

    assert fake.auth_header == f"Token {KEY}"
    assert fake.query["model"] == ["nova-3"]
    assert fake.query["language"] == ["ta"]
    assert fake.query["encoding"] == ["linear16"]
    assert fake.query["sample_rate"] == ["8000"]
    assert fake.received_audio == [b"\x01\x02", b"\x03\x04"]
    assert events == [CommittedTranscript("வணக்கம்")]


async def test_interim_results_become_partial_transcripts():
    fake = FakeDeepgramListen([results("Hel", is_final=False), results("Hello", is_final=False)])
    events = await run(fake, [b"\x00"], language="en")

    assert events == [PartialTranscript("Hel"), PartialTranscript("Hello")]


async def test_finalized_segments_accumulate_until_speech_final_then_commit_once():
    fake = FakeDeepgramListen(
        [
            results("first part", is_final=True, speech_final=False),
            results("second part", is_final=True, speech_final=True),
        ]
    )
    events = await run(fake, [b"\x00"], language="en")

    # The accumulated buffer is surfaced as a partial after each finalized-but-not-final segment too.
    assert events == [PartialTranscript("first part"), CommittedTranscript("first part second part")]


async def test_empty_transcripts_are_ignored():
    fake = FakeDeepgramListen([results("", is_final=False), results("", is_final=True, speech_final=True)])
    events = await run(fake, [b"\x00"], language="en")

    assert events == []


async def test_error_message_raises_asrerror():
    fake = FakeDeepgramListen([{"type": "Error", "variant": "bad_request", "description": "nope"}])
    with pytest.raises(ASRError, match="bad_request"):
        await run(fake, [b"\x00"], language="en")


async def test_metadata_and_speechstarted_are_ignored_not_raised():
    fake = FakeDeepgramListen(
        [
            {"type": "SpeechStarted", "channel": [0], "timestamp": 0.1},
            {"type": "Metadata", "request_id": "x"},
            results("done", is_final=True, speech_final=True),
        ]
    )
    events = await run(fake, [b"\x00"], language="en")

    assert events == [CommittedTranscript("done")]


async def test_connection_failure_is_reported_without_leaking_the_key():
    stt = DeepgramSTT(api_key=KEY, silence_ms=300, connect_timeout_ms=500, url="ws://127.0.0.1:1/v1/listen")
    with pytest.raises(Exception) as exc:
        async with stt.connect(language="en", sample_rate=16000):
            pass
    assert KEY not in str(exc.value)
