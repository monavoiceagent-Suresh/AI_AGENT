"""ASR recovery (app/asr_recovery.py): a call keeps its speech recognition when a provider rejects keyterms or fails.

Simulated, not live: the real Scribe and Deepgram adapters are driven against local WebSocket servers that
behave as the providers are documented to (Scribe: an `invalid_request` message after the handshake; Deepgram:
an `Error` message), and scripted providers cover the recovery plan, audio replay, fallback and its limits.
"""

import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from websockets.asyncio.server import serve

from app import asr_recovery
from app.asr_recovery import Leg, RecoveringTranscriber, recovering_asr
from app.calls import connect_asr
from app.interfaces import CallAudio, CommittedTranscript, PartialTranscript
from app.providers.deepgram_stt import DeepgramSTT
from app.providers.scribe import ASRError, ScribeRealtime
from tests.fakes import FakeLLM
from tests.test_call_reliability import make, wait_for

pytestmark = pytest.mark.anyio

A, B, C = b"\x01\x00" * 160, b"\x02\x00" * 160, b"\x03\x00" * 160  # 20 ms frames at 8 kHz, distinguishable


# ---- scripted providers ------------------------------------------------------------------------------------


def rejected(code: str = "invalid_request") -> ASRError:
    return ASRError(f"scribe {code}: request rejected", code=code, rejected_request=True)


def outage(code: str = "transcriber_error") -> ASRError:
    return ASRError(f"scribe {code}: internal", code=code)


class ScriptedSession:
    def __init__(self, name: str, keyterms: tuple[str, ...] | None):
        self.name = name
        self.keyterms = keyterms
        self.audio = bytearray()
        self.queue: asyncio.Queue = asyncio.Queue()
        self.keyterms_accepted: int | None = None
        self.fail_send = False
        self.closed = False

    async def send_audio(self, pcm: bytes) -> None:
        if self.fail_send or self.closed:
            raise ConnectionError("socket gone")
        self.audio.extend(pcm)

    async def events(self):
        while True:
            item = await self.queue.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def say(self, text: str) -> None:
        self.queue.put_nowait(PartialTranscript(text))
        self.queue.put_nowait(CommittedTranscript(text))

    def fail(self, exc: BaseException) -> None:
        self.queue.put_nowait(exc)


class ScriptedProvider:
    """connect() as scripted: `refuse` (handshake HTTP status, only when keyterms are sent), `down` (every
    connect fails), `gate` (an Event the connect waits on), `on_open` (called with each new session)."""

    def __init__(self, name: str, *, refuse: int | None = None, down: bool = False):
        self.name = name
        self.refuse = refuse
        self.down = down
        self.gate: asyncio.Event | None = None
        self.on_open = None
        self.attempts: list[tuple[str, ...] | None] = []
        self.sessions: list[ScriptedSession] = []

    def connect(self, *, language: str, sample_rate: int, **kwargs):
        terms = kwargs.get("keyterms")
        terms = tuple(terms) if terms is not None else None
        self.attempts.append(terms)

        @asynccontextmanager
        async def session():
            if self.gate is not None:
                await self.gate.wait()
            if self.down:
                raise ConnectionError(f"{self.name} unreachable")
            if terms and self.refuse:
                raise ASRHandshakeRefused(self.refuse)
            s = ScriptedSession(self.name, terms)
            self.sessions.append(s)
            if self.on_open:
                self.on_open(s)
            try:
                yield s
            finally:
                s.closed = True

        return session()


class ASRHandshakeRefused(Exception):
    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.response = SimpleNamespace(status_code=status)


async def collect(transcriber, into: list) -> None:
    async for event in transcriber.events():
        into.append(event)


# ---- post-connect keyterm rejection: same provider, without keyterms, with replay ------------------------


async def test_a_keyterm_rejection_after_connect_reconnects_without_keyterms_and_replays_uncommitted_audio(caplog):
    caplog.set_level(logging.INFO, logger="app.asr_recovery")
    primary, fallback = ScriptedProvider("scribe"), ScriptedProvider("deepgram")
    events: list = []
    async with connect_asr(
        primary,
        fallback,
        language="en",
        sample_rate=8000,
        keyterms=("RFID", "Ananya Rao"),
        provider="elevenlabs_scribe",
        fallback_provider="deepgram_nova3",
    ) as t:
        consumer = asyncio.create_task(collect(t, events))
        first = primary.sessions[0]
        await t.send_audio(A)
        first.say("hello")  # committed: A must never be replayed
        await wait_for(lambda: CommittedTranscript("hello") in events)
        await t.send_audio(B)  # spoken, not yet committed
        first.fail(rejected())
        await wait_for(lambda: len(primary.sessions) == 2 and t.session is primary.sessions[1] and not t._paused)
        await t.send_audio(C)
        second = primary.sessions[1]
        assert primary.attempts == [("RFID", "Ananya Rao"), None]  # the same provider, plain connect
        assert fallback.attempts == []
        assert bytes(second.audio) == B + C  # uncommitted audio replayed, in order, then live audio
        second.say("hello again")
        await wait_for(lambda: CommittedTranscript("hello again") in events)
        consumer.cancel()
    assert [e.text for e in events if isinstance(e, CommittedTranscript)] == ["hello", "hello again"]
    usage = t.usage()
    assert usage["asr_recovery"] == "recovered" and usage["asr_provider"] == "elevenlabs_scribe"
    assert [(s["keyterms_sent"], s["end"]) for s in usage["asr_segments"]] == [(2, "invalid_request"), (0, "closed")]
    assert usage["asr_segments"][1]["replayed_seconds"] == 0.02
    text = caplog.text
    assert "asr_event=keyterms_rejected provider=elevenlabs_scribe" in text and "reason=invalid_request" in text
    assert "asr_event=reconnect_without_keyterms provider=elevenlabs_scribe" in text
    assert "asr_event=audio_replay result=ok provider=elevenlabs_scribe replayed_ms=20" in text
    assert "asr_event=recovery_outcome outcome=recovered" in text
    assert "Ananya" not in text and "RFID" not in text  # terms are never logged


async def test_audio_arriving_during_the_reconnect_follows_the_replay_in_order():
    primary = ScriptedProvider("scribe")
    events: list = []
    async with connect_asr(primary, None, language="en", sample_rate=8000, keyterms=("RFID",)) as t:
        consumer = asyncio.create_task(collect(t, events))
        await t.send_audio(A)
        primary.gate = asyncio.Event()  # the reconnect waits until released
        primary.sessions[0].fail(rejected())
        await wait_for(lambda: len(primary.attempts) == 2)
        await t.send_audio(B)  # while connecting
        await t.send_audio(C)
        primary.gate.set()
        await wait_for(lambda: len(primary.sessions) == 2 and not t._paused)
        assert bytes(primary.sessions[1].audio) == A + B + C
        consumer.cancel()


async def test_the_replay_buffer_is_bounded_and_counts_what_it_dropped(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="app.asr_recovery")
    monkeypatch.setattr(asr_recovery, "REPLAY_SECONDS", 0.05)  # 800 bytes at 8 kHz
    primary = ScriptedProvider("scribe")
    async with connect_asr(primary, None, language="en", sample_rate=8000, keyterms=("RFID",)) as t:
        consumer = asyncio.create_task(collect(t, []))
        for frame in (A, B, C, A, B, C):  # 1920 bytes, never committed
            await t.send_audio(frame)
        primary.sessions[0].fail(rejected())
        await wait_for(lambda: len(primary.sessions) == 2 and not t._paused)
        assert len(primary.sessions[1].audio) == 800
        assert bytes(primary.sessions[1].audio) == (A + B + C + A + B + C)[-800:]  # the newest audio is kept
        consumer.cancel()
    assert "replayed_ms=50 dropped_ms=70" in caplog.text


async def test_scribe_rejecting_keyterms_after_the_handshake_is_recovered_on_the_real_adapter():
    """The real ScribeRealtime adapter against a local server that answers keyterms with invalid_request."""
    connections: list[dict] = []

    async def handler(ws):
        query = parse_qs(urlsplit(ws.request.path).query)
        conn = {"query": query, "audio": bytearray()}
        connections.append(conn)
        config = {"keyterms": query["keyterms"]} if "keyterms" in query else {}
        await ws.send(json.dumps({"message_type": "session_started", "config": config}))
        async for raw in ws:
            conn["audio"].extend(base64.b64decode(json.loads(raw)["audio_base_64"]))
            if "keyterms" in query:
                await ws.send(json.dumps({"message_type": "invalid_request", "error": "keyterm 'Ananya' too long"}))
                return
            if len(conn["audio"]) >= len(A + B):
                await ws.send(json.dumps({"message_type": "committed_transcript", "text": "heard it"}))

    async with serve(handler, "127.0.0.1", 0) as server:
        url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/realtime"
        scribe = ScribeRealtime(
            api_key="k", model_id="scribe_v2_realtime", silence_ms=500, connect_timeout_ms=2000, url=url
        )
        events: list = []
        async with connect_asr(
            scribe, None, language="en", sample_rate=8000, keyterms=("Ananya",), provider="elevenlabs_scribe"
        ) as t:
            consumer = asyncio.create_task(collect(t, events))
            await t.send_audio(A)
            await wait_for(lambda: len(connections) == 2 and not t._paused)
            await t.send_audio(B)
            await wait_for(lambda: CommittedTranscript("heard it") in events)
            consumer.cancel()
            usage = t.usage()
    assert "keyterms" in connections[0]["query"] and "keyterms" not in connections[1]["query"]
    assert bytes(connections[1]["audio"]) == A + B  # A was sent to the rejecting session, then replayed
    assert usage["asr_segments"][0]["keyterms_accepted"] == 1  # session_started echoed it before the rejection
    assert usage["asr_segments"][1]["keyterms_accepted"] == 0  # nothing sent, nothing to accept


async def test_deepgram_rejecting_keyterms_after_the_handshake_is_recovered_on_the_real_adapter():
    connections: list[dict] = []

    async def handler(ws):
        query = parse_qs(urlsplit(ws.request.path).query)
        conn = {"query": query, "audio": bytearray()}
        connections.append(conn)
        async for raw in ws:
            conn["audio"].extend(raw)
            if "keyterm" in query:
                await ws.send(
                    json.dumps({"type": "Error", "variant": "INVALID_QUERY", "description": "Invalid keyterm"})
                )
                return
            if len(conn["audio"]) >= len(A + B):
                await ws.send(
                    json.dumps(
                        {
                            "type": "Results",
                            "is_final": True,
                            "speech_final": True,
                            "channel": {"alternatives": [{"transcript": "heard it"}]},
                        }
                    )
                )

    async with serve(handler, "127.0.0.1", 0) as server:
        url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/listen"
        deepgram = DeepgramSTT(api_key="k", silence_ms=300, connect_timeout_ms=2000, url=url)
        events: list = []
        async with connect_asr(
            deepgram, None, language="hi", sample_rate=8000, keyterms=("GreenFuturz",), provider="deepgram_nova3"
        ) as t:
            consumer = asyncio.create_task(collect(t, events))
            await t.send_audio(A)
            await wait_for(lambda: len(connections) == 2 and not t._paused)
            await t.send_audio(B)
            await wait_for(lambda: CommittedTranscript("heard it") in events)
            consumer.cancel()
            usage = t.usage()
    assert connections[0]["query"]["keyterm"] == ["GreenFuturz"] and "keyterm" not in connections[1]["query"]
    assert bytes(connections[1]["audio"]) == A + B
    assert usage["asr_keyterms_accepted"] is None  # Deepgram never confirms: unknown, not zero
    assert usage["asr_segments"][0]["end"] == "INVALID_QUERY"


async def test_a_deepgram_error_that_does_not_name_keyterms_is_an_outage_not_a_rejection():
    from app.providers.deepgram_stt import _deepgram_error

    assert _deepgram_error({"type": "Error", "description": "Invalid keyterm"}, keyterms_sent=2).rejected_request
    assert not _deepgram_error({"type": "Error", "description": "Invalid keyterm"}, keyterms_sent=0).rejected_request
    assert not _deepgram_error(
        {"type": "Error", "description": "Internal server error"}, keyterms_sent=2
    ).rejected_request


def test_scribe_terms_of_service_errors_are_not_mistaken_for_keyterm_rejections():
    from app.providers.scribe import SCRIBE_REQUEST_REJECTED

    assert SCRIBE_REQUEST_REJECTED == {"invalid_request"}  # unaccepted_terms means terms of service


# ---- fallback ----------------------------------------------------------------------------------------------


async def test_an_outage_after_connect_moves_to_the_fallback_with_its_own_keyterms_and_replays(caplog):
    caplog.set_level(logging.INFO, logger="app.asr_recovery")
    primary, fallback = ScriptedProvider("scribe"), ScriptedProvider("deepgram")
    legs = [
        Leg(primary, "elevenlabs_scribe", "primary", ("RFID",)),
        Leg(fallback, "deepgram_nova3", "fallback", ("RFID", "a much longer product name")),
    ]
    t = RecoveringTranscriber(legs, language="ta", sample_rate=8000)
    async with recovering_asr(t):
        consumer = asyncio.create_task(collect(t, []))
        await t.send_audio(A)
        primary.sessions[0].fail(outage())
        await wait_for(lambda: len(fallback.sessions) == 1 and not t._paused)
        assert primary.attempts == [("RFID",)]  # an outage is not retried without keyterms
        assert fallback.attempts == [("RFID", "a much longer product name")]
        assert bytes(fallback.sessions[0].audio) == A
        consumer.cancel()
    assert t.usage()["asr_provider"] == "deepgram_nova3"
    assert "asr_event=fallback_activated from=elevenlabs_scribe to=deepgram_nova3 keyterms=on" in caplog.text
    assert "asr_event=provider_failed provider=elevenlabs_scribe role=primary reason=transcriber_error" in caplog.text


async def test_when_the_reconnect_without_keyterms_also_fails_the_fallback_takes_over():
    primary, fallback = ScriptedProvider("scribe"), ScriptedProvider("deepgram")
    async with connect_asr(primary, fallback, language="en", sample_rate=8000, keyterms=("UHF",)) as t:
        consumer = asyncio.create_task(collect(t, []))
        await t.send_audio(A)
        primary.down = True
        primary.sessions[0].fail(rejected())
        await wait_for(lambda: len(fallback.sessions) == 1 and not t._paused)
        assert primary.attempts == [("UHF",), None]
        assert fallback.attempts == [("UHF",)]
        assert bytes(fallback.sessions[0].audio) == A
        consumer.cancel()
    assert t.outcome == "recovered"


async def test_a_handshake_refusal_of_keyterms_still_reconnects_without_them():
    primary, fallback = ScriptedProvider("scribe", refuse=414), ScriptedProvider("deepgram")
    async with connect_asr(primary, fallback, language="te", sample_rate=8000, keyterms=("UHF",)) as t:
        assert t.session.name == "scribe"
    assert primary.attempts == [("UHF",), None] and fallback.attempts == []


async def test_a_failed_send_recovers_even_if_the_provider_stream_stays_open():
    primary, fallback = ScriptedProvider("scribe"), ScriptedProvider("deepgram")
    async with connect_asr(primary, fallback, language="en", sample_rate=8000) as t:
        consumer = asyncio.create_task(collect(t, []))
        await t.send_audio(A)
        primary.sessions[0].fail_send = True
        await t.send_audio(B)  # never raises into the call; the session is closed and replaced
        await wait_for(lambda: len(fallback.sessions) == 1 and not t._paused)
        assert bytes(fallback.sessions[0].audio) == A + B
        consumer.cancel()


async def test_a_stream_that_ends_by_itself_mid_call_is_recovered():
    primary, fallback = ScriptedProvider("scribe"), ScriptedProvider("deepgram")
    async with connect_asr(primary, fallback, language="en", sample_rate=8000) as t:
        consumer = asyncio.create_task(collect(t, []))
        primary.sessions[0].queue.put_nowait(None)  # the provider closed the stream without an error
        await wait_for(lambda: len(fallback.sessions) == 1 and not t._paused)
        consumer.cancel()
    assert t.usage()["asr_segments"][0]["end"] == "stream_closed"


# ---- bounded: every configuration at most once, then a controlled failure ---------------------------------


async def test_every_configuration_is_tried_at_most_once_then_the_error_reaches_the_call(caplog):
    caplog.set_level(logging.INFO, logger="app.asr_recovery")
    primary, fallback = ScriptedProvider("scribe"), ScriptedProvider("deepgram")
    for provider in (primary, fallback):
        provider.on_open = lambda s: s.fail(rejected())  # every session is rejected straight away
    async with connect_asr(primary, fallback, language="en", sample_rate=8000, keyterms=("UHF",)) as t:
        with pytest.raises(ASRError):
            await asyncio.wait_for(collect(t, []), 5)
    assert primary.attempts == [("UHF",), None]
    assert fallback.attempts == [("UHF",), None]
    assert t.outcome == "failed"
    assert "asr_event=recovery_outcome outcome=failed reason=invalid_request configurations_tried=4" in caplog.text


async def test_without_keyterms_or_fallback_a_mid_call_failure_ends_recognition_without_retrying():
    primary = ScriptedProvider("scribe")
    async with connect_asr(primary, None, language="en", sample_rate=8000) as t:
        primary.sessions[0].fail(outage())
        with pytest.raises(ASRError, match="transcriber_error"):
            await asyncio.wait_for(collect(t, []), 5)
    assert primary.attempts == [None]


async def test_a_call_survives_a_keyterm_rejection_with_its_conversation_intact():
    primary = ScriptedProvider("scribe")
    t = RecoveringTranscriber(
        [Leg(primary, "elevenlabs_scribe", "primary", ("Sunrise",))], language="en", sample_rate=8000
    )
    async with recovering_asr(t):
        session, media, _ = make(FakeLLM(["We open at nine.", "Yes, Saturdays too."]), asr=t)
        task = asyncio.create_task(session.run())
        await wait_for(lambda: len(session.turns) == 1, timeout=8)  # the greeting
        media.inbox.put_nowait(CallAudio(A))
        primary.sessions[0].say("When do you open")
        await wait_for(lambda: len(session.turns) == 2, timeout=8)
        primary.sessions[0].fail(rejected())
        await wait_for(lambda: len(primary.sessions) == 2 and not t._paused, timeout=8)
        primary.sessions[1].say("Saturdays too")
        await wait_for(lambda: len(session.turns) == 3, timeout=8)
        media.hang_up()
        assert await asyncio.wait_for(task, 8) == "caller:callended"
    assert [turn["user_text"] for turn in session.turns] == [None, "When do you open", "Saturdays too"]
    assert [m["content"] for m in session.history if m["role"] == "user"] == ["When do you open", "Saturdays too"]


async def test_a_call_whose_recognition_cannot_be_recovered_ends_with_a_recorded_reason():
    primary = ScriptedProvider("scribe")
    t = RecoveringTranscriber([Leg(primary, "elevenlabs_scribe", "primary", ())], language="en", sample_rate=8000)
    async with recovering_asr(t):
        session, media, _ = make(FakeLLM([]), asr=t)
        task = asyncio.create_task(session.run())
        await wait_for(lambda: len(session.turns) == 1, timeout=8)
        primary.sessions[0].fail(outage())
        assert await asyncio.wait_for(task, 8) == "asr_failed:ASRError"


# ---- isolation: concurrent calls each keep their own keyterms --------------------------------------------


async def test_concurrent_recovering_calls_on_one_provider_keep_their_own_keyterms():
    shared = ScriptedProvider("scribe")

    async def call(terms: tuple[str, ...]) -> tuple:
        async with connect_asr(shared, None, language="en", sample_rate=8000, keyterms=terms) as t:
            await asyncio.sleep(0.01)
            return t.session.keyterms

    results = await asyncio.gather(call(("Acme", "RFID")), call(("Zenith",)), call(()))
    assert results == [("Acme", "RFID"), ("Zenith",), None]
