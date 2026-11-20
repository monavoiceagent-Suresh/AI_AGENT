"""Speech-recognition prompting: a persona's vocabulary reaches the ASR provider as native keyterms, per call.

Covers parsing and each provider's own limits (app/recognition.py, with tests/fixtures/recognition_cases.json
shared with Mona's preview), both provider adapters' handshake query, connect_asr's safe fallback when a provider
refuses keyterms at the handshake (post-connect rejections: tests/test_asr_recovery.py), persona JSON propagation
for inbound routes and outbound jobs, and run_call: each concurrent call connects with its own persona's
keyterms and nothing reaches the LLM prompt.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import ValidationError
from websockets.asyncio.server import serve

from app import calls, recognition
from app.calls import connect_asr
from app.models import Agent, PersonaIn
from app.providers.deepgram_stt import DeepgramSTT
from app.providers.scribe import ASRError, ScribeRealtime
from app.recognition import (
    DEEPGRAM_LIMITS,
    MAX_HINTS_CHARS,
    SCRIBE_LIMITS,
    STRICTEST_LIMITS,
    keyterms,
    limits_for,
)
from tests.fakes import make_settings

CASES = json.loads((Path(__file__).parent / "fixtures" / "recognition_cases.json").read_text(encoding="utf-8"))

pytestmark = pytest.mark.anyio


# ---- parsing and provider limits --------------------------------------------------------------------------


def test_empty_or_missing_hints_give_no_keyterms():
    assert keyterms("") == ()
    assert keyterms(None) == ()
    assert keyterms(" , ;\n\n ,") == ()


def test_splits_on_commas_semicolons_and_lines_keeping_multi_word_terms():
    assert keyterms("RFID, UHF; EPC Gen2\nGreenFuturz\r\nDr. Ananya Rao") == (
        "RFID",
        "UHF",
        "EPC Gen2",
        "GreenFuturz",
        "Dr. Ananya Rao",
    )


def test_terms_with_unsupported_characters_are_dropped_whole_never_cleaned_up():
    assert keyterms("<b>Mona</b>, {Acme}, [Pro], back\\slash, ok") == ("ok",)
    assert keyterms("\x07bell, nul\x00, fine") == ("fine",)
    assert keyterms("tab\there,  nb\u00a0sp ") == ("tab here", "nb sp")  # whitespace is only collapsed


def test_case_insensitive_duplicates_are_dropped_keeping_the_first():
    assert keyterms("RFID, rfid, Rfid, UHF") == ("RFID", "UHF")


def test_scribe_realtime_limits_are_50_terms_of_20_characters_and_5_words():
    assert (SCRIBE_LIMITS.max_terms, SCRIBE_LIMITS.max_term_chars, SCRIBE_LIMITS.max_term_words) == (50, 20, 5)
    assert limits_for("elevenlabs_scribe") is SCRIBE_LIMITS


@pytest.mark.parametrize(
    ("length", "scribe_keeps", "deepgram_keeps"),
    [(20, True, True), (21, False, True), (50, False, True), (51, False, False), (500, False, False)],
)
def test_per_term_length_boundaries_are_provider_specific(length, scribe_keeps, deepgram_keeps):
    term = "x" * length
    assert keyterms(f"{term}, ok", "elevenlabs_scribe") == ((term,) if scribe_keeps else ()) + ("ok",)
    assert keyterms(f"{term}, ok", "deepgram_nova3") == ((term,) if deepgram_keeps else ()) + ("ok",)


def test_terms_are_dropped_never_truncated():
    assert keyterms("abcdefghijklmnopqrstu", "elevenlabs_scribe") == ()  # not "abcdefghijklmnopqrst"
    assert keyterms("x" * 51, "deepgram_nova3") == ()


def test_a_sixth_word_drops_the_term():
    assert keyterms("a b c d e, one two three four five six") == ("a b c d e",)


@pytest.mark.parametrize("provider", ["elevenlabs_scribe", "deepgram_nova3"])
def test_term_count_is_capped_at_50(provider):
    terms = keyterms(", ".join(f"t{i}" for i in range(70)), provider)
    assert len(terms) == 50 and terms[0] == "t0" and terms[-1] == "t49"


def test_deepgram_alone_has_a_total_budget_for_its_500_token_limit():
    hints = ", ".join(f"{i:02d}" + "a" * 38 for i in range(20))  # 20 terms of 40 characters
    assert sum(map(len, keyterms(hints, "deepgram_nova3"))) <= DEEPGRAM_LIMITS.max_total_chars
    assert len(keyterms(hints, "deepgram_nova3")) == 12
    assert keyterms(hints, "elevenlabs_scribe") == ()  # 40 characters is over Scribe's per-term limit


def test_an_unknown_provider_gets_the_strictest_limits():
    assert limits_for("something_new") is STRICTEST_LIMITS
    assert keyterms("x" * 21, "something_new") == ()


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_shared_cases_match_the_dashboard_preview(case):
    """The same cases Mona's preview is tested against (recognition-hints.test.ts): one normalisation."""
    for provider in ("elevenlabs_scribe", "deepgram_nova3"):
        assert list(keyterms(case["hints"], provider)) == case[provider]


def test_non_latin_scripts_are_kept():
    assert keyterms("சென்னை, हैदराबाद, విజయవాడ") == ("சென்னை", "हैदराबाद", "విజయవాడ")


def test_persona_accepts_recognition_hints_and_bounds_their_size():
    base = {
        "key": "front-desk",
        "name": "Front desk",
        "language": "en",
        "instructions": "Be helpful.",
        "greeting": "Hello.",
        "version": 1,
    }
    assert PersonaIn.model_validate(base).recognition_hints == ""  # older dashboards omit it
    assert PersonaIn.model_validate(base | {"recognitionHints": "RFID, UHF"}).recognition_hints == "RFID, UHF"
    with pytest.raises(ValidationError):
        PersonaIn.model_validate(base | {"recognitionHints": "x" * (MAX_HINTS_CHARS + 1)})


# ---- provider adapters: the handshake query ---------------------------------------------------------------


class QueryRecorder:
    """A local WebSocket server that records each handshake's query and then closes."""

    def __init__(self, first_message: dict | None = None):
        self.queries: list[dict[str, list[str]]] = []
        self.first_message = first_message

    async def handler(self, ws):
        self.queries.append(parse_qs(urlsplit(ws.request.path).query))
        if self.first_message:
            await ws.send(json.dumps(self.first_message))
        await asyncio.sleep(0.05)


@asynccontextmanager
async def local_server(recorder: QueryRecorder):
    async with serve(recorder.handler, "127.0.0.1", 0) as server:
        yield f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}/listen"


def scribe(url: str) -> ScribeRealtime:
    return ScribeRealtime(api_key="k", model_id="scribe_v2_realtime", silence_ms=500, connect_timeout_ms=2000, url=url)


async def test_scribe_sends_each_keyterm_as_a_repeated_keyterms_parameter():
    recorder = QueryRecorder()
    async with local_server(recorder) as url:
        async with scribe(url).connect(language="en", sample_rate=8000, keyterms=("RFID", "EPC Gen2")):
            pass
    assert recorder.queries[0]["keyterms"] == ["RFID", "EPC Gen2"]
    assert recorder.queries[0]["model_id"] == ["scribe_v2_realtime"]


async def test_scribe_without_keyterms_sends_exactly_the_old_query():
    recorder = QueryRecorder()
    async with local_server(recorder) as url:
        async with scribe(url).connect(language="ta", sample_rate=8000):
            pass
    assert set(recorder.queries[0]) == {
        "model_id",
        "audio_format",
        "language_code",
        "commit_strategy",
        "vad_silence_threshold_secs",
    }


async def test_scribe_logs_how_many_keyterms_the_server_accepted_never_the_terms(caplog):
    recorder = QueryRecorder({"message_type": "session_started", "config": {"keyterms": ["RFID", "Ananya"]}})
    caplog.set_level("INFO", logger="app.providers.scribe")
    async with local_server(recorder) as url:
        async with scribe(url).connect(language="en", sample_rate=8000, keyterms=("RFID", "Ananya")) as session:
            with pytest.raises(ASRError):  # the fake server then closes, and reopening is refused
                async for _ in session.events():
                    pass
    started = [r.getMessage() for r in caplog.records if "session started" in r.getMessage()]
    assert started and "keyterms_sent=2 keyterms_accepted=2" in started[0]
    assert "Ananya" not in caplog.text


async def test_deepgram_sends_each_keyterm_as_a_repeated_keyterm_parameter():
    recorder = QueryRecorder()
    async with local_server(recorder) as url:
        stt = DeepgramSTT(api_key="k", silence_ms=300, connect_timeout_ms=2000, url=url)
        async with stt.connect(language="hi", sample_rate=8000, keyterms=("UHF", "GreenFuturz")):
            pass
        async with stt.connect(language="hi", sample_rate=8000):
            pass
    assert recorder.queries[0]["keyterm"] == ["UHF", "GreenFuturz"]
    assert "keyterm" not in recorder.queries[1] and "keyterms" not in recorder.queries[1]


async def test_concurrent_sessions_on_one_adapter_each_send_their_own_keyterms():
    recorder = QueryRecorder()
    async with local_server(recorder) as url:
        adapter = scribe(url)

        async def call(terms: tuple[str, ...]) -> None:
            async with adapter.connect(language="en", sample_rate=8000, keyterms=terms):
                await asyncio.sleep(0.01)

        await asyncio.gather(call(("Acme", "RFID")), call(("Zenith",)), call(()))
    sent = sorted(tuple(q.get("keyterms", [])) for q in recorder.queries)
    assert sent == [(), ("Acme", "RFID"), ("Zenith",)]


# ---- connect_asr: keyterms forwarded, refused keyterms never cost a call its ASR -------------------------


class Refused(Exception):
    """Shaped like websockets' InvalidStatus: the handshake was answered with an HTTP error status."""

    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.response = SimpleNamespace(status_code=status)


class ScriptedASR:
    def __init__(self, name: str, *, refuse_keyterms_with: int | None = None, down: bool = False):
        self.name = name
        self.refuse_keyterms_with = refuse_keyterms_with
        self.down = down
        self.attempts: list[tuple[str, ...] | None] = []  # None = called without the keyterms argument

    def connect(self, *, language: str, sample_rate: int, **kwargs):
        terms = kwargs.get("keyterms")
        self.attempts.append(tuple(terms) if terms is not None else None)

        @asynccontextmanager
        async def session():
            if self.down:
                raise ConnectionError(f"{self.name} unreachable")
            if terms and self.refuse_keyterms_with:
                raise Refused(self.refuse_keyterms_with)
            yield SimpleNamespace(name=self.name)

        return session()


async def test_keyterms_reach_the_primary_provider():
    primary = ScriptedASR("primary")
    async with connect_asr(primary, None, language="en", sample_rate=8000, keyterms=("RFID",)) as t:
        assert t.session.name == "primary"
    assert primary.attempts == [("RFID",)]


async def test_no_keyterms_connects_exactly_as_before_without_the_argument():
    primary = ScriptedASR("primary")
    async with connect_asr(primary, None, language="en", sample_rate=8000):
        pass
    async with connect_asr(primary, None, language="en", sample_rate=8000, keyterms=()):
        pass
    assert primary.attempts == [None, None]


@pytest.mark.parametrize("status", [400, 413, 414, 422])
async def test_a_refused_keyterm_request_retries_the_same_provider_without_keyterms(status):
    primary, fallback = ScriptedASR("primary", refuse_keyterms_with=status), ScriptedASR("fallback")
    async with connect_asr(primary, fallback, language="te", sample_rate=8000, keyterms=("UHF",)) as t:
        assert t.session.name == "primary"
    assert primary.attempts == [("UHF",), None]
    assert fallback.attempts == []


async def test_an_outage_goes_straight_to_the_fallback_which_also_gets_the_keyterms():
    primary, fallback = ScriptedASR("primary", down=True), ScriptedASR("fallback")
    async with connect_asr(primary, fallback, language="en", sample_rate=8000, keyterms=("UHF",)) as t:
        assert t.session.name == "fallback"
    assert primary.attempts == [("UHF",)]
    assert fallback.attempts == [("UHF",)]


async def test_an_auth_failure_is_not_retried_without_keyterms():
    primary = ScriptedASR("primary", refuse_keyterms_with=401)
    with pytest.raises(Refused):
        async with connect_asr(primary, None, language="en", sample_rate=8000, keyterms=("UHF",)):
            pass
    assert primary.attempts == [("UHF",)]


# ---- persona propagation and the call itself ------------------------------------------------------------


def stored_agent(**overrides) -> Agent:
    now = datetime.now(UTC)
    fields = {
        "id": 7,
        "created_at": now,
        "updated_at": now,
        "name": "Base",
        "language": "en",
        "instructions": "Base instructions.",
        "greeting": "Hello.",
        "client_id": "client-a",
        "persona_key": "front-desk",
    }
    return Agent(**(fields | overrides))


def persona_json(hints: str | None, **overrides) -> dict:
    """A persona exactly as stored on an inbound route or outbound job (PersonaIn.model_dump())."""
    persona = PersonaIn(
        key="front-desk",
        name="Front desk",
        language="en",
        instructions="Answer questions.",
        greeting="Hello.",
        version=3,
        recognition_hints=hints or "",
        **overrides,
    ).model_dump()
    if hints is None:
        del persona["recognition_hints"]  # a route or job published before this field existed
    return persona


def test_with_persona_carries_the_hints_and_old_snapshots_have_none():
    assert calls.with_persona(stored_agent(), persona_json("RFID, UHF")).recognition_hints == "RFID, UHF"
    assert calls.with_persona(stored_agent(recognition_hints="stale"), persona_json(None)).recognition_hints == ""


def test_hints_never_reach_the_llm_prompt_or_knowledge():
    agent = calls.with_persona(stored_agent(), persona_json("GreenFuturz, EPC Gen2"))
    config = calls.session_config(agent, make_settings(), "Inbound call; the caller's name is not known.")
    assert "GreenFuturz" not in config.system_prompt
    assert agent.knowledge == ""


class CallHarness:
    """run_call with its I/O replaced: records the keyterms each call's ASR providers were prepared with."""

    def __init__(self, monkeypatch):
        self.connected: dict[str, tuple[str, ...]] = {}
        self.legs: dict[str, dict[str, tuple[str, ...]]] = {}
        harness = self

        @asynccontextmanager
        async def recording_recovering_asr(transcriber):
            await asyncio.sleep(0.01)  # let concurrent calls interleave
            name = asyncio.current_task().get_name()
            harness.legs[name] = {leg.provider: leg.keyterms for leg in transcriber._legs}
            harness.connected[name] = transcriber._legs[0].keyterms
            raise RuntimeError("test ends the call here")
            yield  # pragma: no cover

        async def start_call(pool, **kwargs):
            return 1

        async def finalize(*args, **kwargs):
            return None

        monkeypatch.setattr(calls, "recovering_asr", recording_recovering_asr)
        monkeypatch.setattr(calls.db, "start_call", start_call)
        monkeypatch.setattr(calls, "_finalize", finalize)
        monkeypatch.setattr(calls, "_spawn", lambda coro: coro.close())
        monkeypatch.setattr(calls, "resolve_tts", lambda svc, agent: (object(), "bakbak", None, None))
        monkeypatch.setattr(
            calls, "resolve_asr", lambda svc, agent: (object(), "elevenlabs_scribe", object(), "deepgram_nova3")
        )
        monkeypatch.setattr(calls, "resolve_llm", lambda svc, agent: object())

        async def mark_connected(call_sid):
            return None

        self.svc = SimpleNamespace(
            settings=make_settings(),
            pool=None,
            allocator=SimpleNamespace(mark_connected_by_call_sid=mark_connected),
        )

    async def run(self, agent: Agent, call_sid: str, caller_context: str) -> None:
        async def close():
            return None

        media = SimpleNamespace(sample_rate=8000, close=close)
        start = SimpleNamespace(
            call_sid=call_sid, stream_sid="s", from_number="+911", to_number="+912", called_number=None
        )
        task = asyncio.current_task()
        task.set_name(call_sid)
        await calls.run_call(self.svc, media, start, agent, None, caller_context)


async def test_inbound_and_outbound_calls_connect_with_their_own_personas_keyterms(monkeypatch):
    harness = CallHarness(monkeypatch)
    inbound = calls.with_persona(stored_agent(), persona_json("RFID, UHF"))  # a published route's persona
    outbound = calls.with_persona(  # a campaign job's persona, from another tenant
        stored_agent(client_id="client-b"), persona_json("Dr. Ananya Rao; cardiology")
    )
    legacy = calls.with_persona(stored_agent(), persona_json(None))  # published before hints existed

    await asyncio.gather(  # each coroutine runs in its own task, like each live call
        harness.run(inbound, "CA-in", calls.prompts.caller_context("inbound", None)),
        harness.run(outbound, "CA-out", calls.prompts.caller_context("outbound", "Ravi")),
        harness.run(legacy, "CA-old", calls.prompts.caller_context("inbound", None)),
    )
    assert harness.connected == {
        "CA-in": ("RFID", "UHF"),
        "CA-out": ("Dr. Ananya Rao", "cardiology"),
        "CA-old": (),
    }


async def test_each_provider_gets_the_personas_terms_cut_to_its_own_limits(monkeypatch):
    harness = CallHarness(monkeypatch)
    agent = calls.with_persona(stored_agent(), persona_json("RFID, Prestige Lakeside Whitefield"))  # 2nd: 28 chars
    await harness.run(agent, "CA-1", calls.prompts.caller_context("inbound", None))
    assert harness.legs["CA-1"] == {
        "elevenlabs_scribe": ("RFID",),
        "deepgram_nova3": ("RFID", "Prestige Lakeside Whitefield"),
    }


async def test_unsupported_language_keeps_transcribing_without_hints(monkeypatch):
    """A provider that cannot bias this language/model refuses the handshake; the call still gets ASR."""
    primary = ScriptedASR("primary", refuse_keyterms_with=400)
    terms = recognition.keyterms("ஆர்எஃப்ஐடி, RFID")
    async with connect_asr(primary, None, language="ta", sample_rate=8000, keyterms=terms) as t:
        assert t.session.name == "primary"
    assert primary.attempts == [terms, None]
