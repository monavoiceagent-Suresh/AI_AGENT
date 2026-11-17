"""P0 call lifecycle outside the conversation itself: ASR reconnects, Exotel termination verification, traceable
stream failures, call initiation validation, inbound routing without a silent global default, and LLM model
validation for the Super Admin form. No database or network: collaborators are faked at their boundary."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError
from websockets.exceptions import ConnectionClosedError

from app import api, calls, db, dispatch, dispatch_db, inbound_db, llm_models
from app.calls import CallRejected
from app.interfaces import CommittedTranscript
from app.models import DispatchIn
from app.providers.exotel import ExotelCallRecord, StreamStart
from app.providers.scribe import ASRError, ScribeSession
from tests.fakes import make_settings

pytestmark = pytest.mark.anyio


# ---- ASR: a Scribe socket that drops mid-call is reopened, bounded ---------------------------------------


class FakeSocket:
    """A Scribe websocket: yields its scripted messages, then drops (or stays open forever)."""

    def __init__(self, messages: list[dict], *, drop: bool = True) -> None:
        self._messages = [json.dumps(m) for m in messages]
        self._drop = drop
        self.sent: list[dict] = []
        self.closed = False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for message in self._messages:
            yield message
        if self._drop:
            raise ConnectionClosedError(None, None)
        await asyncio.Event().wait()

    async def send(self, frame: str) -> None:
        if self.closed:
            raise ConnectionClosedError(None, None)
        self.sent.append(json.loads(frame))

    async def close(self) -> None:
        self.closed = True


async def test_scribe_reconnects_after_a_drop_and_replays_the_audio_from_the_gap():
    first = FakeSocket([{"message_type": "committed_transcript", "text": "hello"}])
    second = FakeSocket([{"message_type": "committed_transcript", "text": "are you there"}], drop=False)
    gate = asyncio.Event()

    async def reopen():
        await gate.wait()
        return second

    session = ScribeSession(first, 8000, reopen=reopen)
    events = session.events()
    assert await anext(events) == CommittedTranscript("hello")
    pending = asyncio.ensure_future(anext(events))  # notices the drop and starts reopening
    await asyncio.sleep(0.01)
    await session.send_audio(bytes(320))  # caller audio during the gap is held, not lost or raised
    gate.set()

    assert await asyncio.wait_for(pending, 2) == CommittedTranscript("are you there")
    assert session.reconnects == 1
    assert second.sent and second.sent[0]["message_type"] == "input_audio_chunk"  # the held audio was replayed


async def test_scribe_gives_up_after_its_reconnect_budget_with_a_clear_error():
    sockets = [FakeSocket([]) for _ in range(3)]

    async def reopen():
        return sockets.pop(0)

    session = ScribeSession(FakeSocket([]), 8000, reopen=reopen, max_reconnects=2)
    with pytest.raises(ASRError, match="could not be reopened"):
        async for _ in session.events():
            pass
    assert session.reconnects == 2


# ---- agent hang-up: verify with Exotel that the call really ended -----------------------------------------


class RecordExotel:
    def __init__(self, status: str | Exception) -> None:
        self.status = status

    async def get_call_record(self, call_sid: str, *, wait_seconds: float) -> ExotelCallRecord:
        if isinstance(self.status, Exception):
            raise self.status
        return ExotelCallRecord(to="", phone_number_sid="", direction="outbound-api", forwarded_from="",
                                status=self.status)


@pytest.mark.parametrize(
    ("status", "verdict"),
    [("completed", "terminated"), ("in-progress", "still_live"), (httpx.ConnectError("down"), "unverified")],
)
async def test_agent_hangup_is_verified_against_exotels_record(status, verdict, caplog):
    svc = SimpleNamespace(settings=make_settings(), exotel=RecordExotel(status))
    assert await calls.verify_exotel_termination(svc, "CA1", 5, "agent_ended", delay=0) == verdict
    if verdict == "still_live":
        assert any("still live" in r.getMessage() for r in caplog.records)


# ---- a connected call whose media stream fails is traceable, not left "initiated" -------------------------


class StatusPool:
    def __init__(self, result: str) -> None:
        self.result = result
        self.queries: list[tuple] = []

    async def execute(self, query: str, *args):
        self.queries.append((query, args))
        return self.result


async def test_a_stream_that_cannot_start_is_recorded_on_its_call_row():
    pool = StatusPool("UPDATE 1")
    await api._record_stream_failure(SimpleNamespace(pool=pool), "CA7", "stream_failed:CallRejected")

    query, args = pool.queries[0]
    assert "status = 'stream_failed'" in query and "status = 'initiated'" in query
    assert args == ("CA7", "stream_failed:CallRejected")


async def test_exotel_status_callbacks_keep_their_own_terminal_reason():
    pool = StatusPool("UPDATE 1")
    assert await db.apply_status_callback(pool, "CA8", "no_answer")
    assert pool.queries[0][1] == ("CA8", "no_answer", "exotel:no_answer")


# ---- call initiation ---------------------------------------------------------------------------------------


def dispatch_body(**overrides) -> dict:
    body = {
        "idempotencyKey": "key-12345678",
        "campaignId": "c1",
        "clientId": "client-1",
        "persona": {"key": "shanthi", "name": "Shanthi", "language": "en", "instructions": "i", "greeting": "Hi",
                    "version": 1},
        "callingWindow": {"start": "09:00", "end": "18:00"},
        "retryPolicy": {"maxTries": 2, "retryGapMinutes": 30, "callbackWindowDays": 1},
        "leads": [{"id": "l1", "phone": "+919876543210", "name": "A"}],
        "webhookUrl": "https://mona.example/api/webhooks/voice-engine",
    }
    return body | overrides


def test_a_valid_dispatch_payload_is_accepted_by_the_contract():
    assert DispatchIn.model_validate(dispatch_body()).leads[0].phone == "+919876543210"


@pytest.mark.parametrize(
    "overrides",
    [
        {"leads": []},
        {"leads": [{"id": "l1", "phone": "12"}]},
        {"webhookUrl": "ftp://x"},
        {"persona": {"key": "x"}},
        {"callingWindow": {"start": "18:00", "end": "09:00"}},
        {"callerId": "08047491899"},
    ],
)
def test_an_invalid_dispatch_payload_is_rejected_before_any_call(overrides):
    with pytest.raises(ValidationError):
        DispatchIn.model_validate(dispatch_body(**overrides))


@pytest.mark.parametrize(
    ("phone", "problem"),
    [("+919876543210", None), ("09876543210", None), ("+9109876543210", None), ("+91987654321", "exactly ten"),
     ("12345", "not a dialable")],
)
def test_lead_numbers_are_checked_before_dialling(phone, problem):
    found = dispatch.lead_number_problem(phone)
    assert (found is None) if problem is None else (problem in found)


async def test_an_undialable_lead_is_failed_as_invalid_number_without_calling_exotel(monkeypatch):
    lead = {"lead_pk": 1, "lead_id": "l1", "phone": "+91987654321", "attempts": 0, "max_tries": 3, "job_id": "j1",
            "job_status": "running", "retry_gap_minutes": 30, "agent_id": 3, "caller_id": None}
    settled: list[dict] = []

    async def next_due_lead(pool, enforcing):
        return lead

    async def claim_lead(pool, lead_pk):
        return True

    async def settle_lead(pool, **kwargs):
        settled.append(kwargs)
        return True

    async def no_dial(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("an invalid number was dialled")

    monkeypatch.setattr(dispatch_db, "next_due_lead", next_due_lead)
    monkeypatch.setattr(dispatch_db, "claim_lead", claim_lead)
    monkeypatch.setattr(dispatch_db, "settle_lead", settle_lead)
    monkeypatch.setattr(calls, "place_outbound_call", no_dial)
    svc = SimpleNamespace(settings=make_settings(enable_outbound_calls=True), pool=None)

    assert await dispatch.place_next_call(svc) is True
    (result,) = settled
    assert result["status"] == "failed" and result["outcome"] == "Invalid number"  # final: no pointless retry
    assert result["event"]["eventType"] == "call.failed"
    assert "ten digits" in result["event"]["data"]["failureReason"]


async def test_an_exotel_rejection_reaches_mona_with_its_reason(monkeypatch):
    from app.providers.exotel import ExotelError

    lead = {"lead_pk": 1, "lead_id": "l1", "phone": "+919876543210", "attempts": 0, "max_tries": 3, "job_id": "j1",
            "job_status": "running", "retry_gap_minutes": 30, "agent_id": 3, "caller_id": None, "name": None,
            "voice_id": None, "tts_model_id": None}
    settled: list[dict] = []

    async def fake(*args, **kwargs):
        return lead if "enforcing" in kwargs else True

    async def settle_lead(pool, **kwargs):
        settled.append(kwargs)
        return True

    async def get_agent(pool, agent_id):
        return {"id": 3, "name": "A", "language": "en", "instructions": "i", "greeting": "g",
                "created_at": "2026-11-17T00:00:00Z", "updated_at": "2026-11-17T00:00:00Z"}

    async def rejected(*args, **kwargs):
        raise ExotelError("Exotel connect failed (400): Invalid From number")

    monkeypatch.setattr(dispatch_db, "next_due_lead", fake)
    monkeypatch.setattr(dispatch_db, "claim_lead", fake)
    monkeypatch.setattr(dispatch_db, "settle_lead", settle_lead)
    monkeypatch.setattr(db, "get_agent", get_agent)
    monkeypatch.setattr(calls, "place_outbound_call", rejected)
    svc = SimpleNamespace(settings=make_settings(enable_outbound_calls=True), pool=None)

    assert await dispatch.place_next_call(svc) is True
    data = settled[0]["event"]["data"]
    assert data["outcome"] == "dial_failed" and "Invalid From number" in data["failureReason"]
    assert data["leadState"] == "retry"


# ---- inbound routing never silently falls back to a global default agent ----------------------------------


async def test_an_unrouted_inbound_number_is_refused_by_default(monkeypatch):
    noted: list[tuple] = []

    async def no_route(pool, phone):
        return None

    async def note_unrouted(pool, phone, raw, answered_by):
        noted.append((phone, answered_by))

    async def default_agent(pool):  # pragma: no cover - must not be consulted when the default is off
        raise AssertionError("the global default agent was consulted")

    monkeypatch.setattr(inbound_db, "find_route_for_call", no_route)
    monkeypatch.setattr(inbound_db, "note_unrouted", note_unrouted)
    monkeypatch.setattr(db, "get_inbound_default_agent", default_agent)
    settings = make_settings(exotel_resolve_called_number=False)
    assert settings.inbound_legacy_default_enabled is False
    start = StreamStart(stream_sid="S1", call_sid="CA1", from_number="+919800000000", to_number="08047491899",
                        sample_rate=16000)

    with pytest.raises(CallRejected, match="No inbound route"):
        await calls._resolve_inbound(SimpleNamespace(settings=settings, pool=None), start)
    assert noted == [("918047491899", "refused")]  # dialled-number normalization: 0-prefixed -> 91...


async def test_a_routed_inbound_number_answers_with_its_own_persona(monkeypatch):
    persona = {"key": "front-desk", "version": 4, "name": "Desk", "language": "en", "workflow": "general",
               "instructions": "i", "greeting": "Hello, Acme here.", "knowledge": ""}
    route = {"phone": "+918047491899", "active": True, "agent_id": 11, "version": 2, "persona": persona}
    looked_up: list[str] = []

    async def find_route(pool, phone):
        looked_up.append(phone)
        return route

    async def get_agent(pool, agent_id):
        return {"id": agent_id, "name": "A", "language": "en", "instructions": "x", "greeting": "x",
                "client_id": "client-1", "created_at": "2026-11-17T00:00:00Z", "updated_at": "2026-11-17T00:00:00Z"}

    class Allocator:
        async def reserve(self, *args, **kwargs):
            return True

    async def known(*args, **kwargs):
        return None

    monkeypatch.setattr(inbound_db, "find_route_for_call", find_route)
    monkeypatch.setattr(db, "get_agent", get_agent)
    from app import tenants

    monkeypatch.setattr(tenants, "require_known", known)
    svc = SimpleNamespace(settings=make_settings(), pool=None, allocator=Allocator())
    start = StreamStart(stream_sid="S1", call_sid="CA1", from_number="+919800000000", to_number="+91 80 4749 1899",
                        sample_rate=16000)

    resolved = await calls._resolve_inbound(svc, start)
    assert looked_up == ["918047491899"]
    assert resolved.agent.greeting == "Hello, Acme here."  # the route's own persona, not a default agent's
    assert resolved.agent.id == 11 and resolved.call_id is None
    assert resolved.start.called_number == "+918047491899"


# ---- Super Admin LLM model validation ----------------------------------------------------------------------


CATALOGUE = {
    "data": [
        {"id": "google/gemini-2.5-flash-lite", "name": "Gemini Flash Lite",
         "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
         "supported_parameters": ["max_tokens", "temperature", "tools"]},
        {"id": "openai/gpt-4o-mini", "name": "GPT-4o mini",
         "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
         "supported_parameters": ["max_tokens", "temperature", "tools"]},
        {"id": "some/image-model", "name": "Image",
         "architecture": {"input_modalities": ["text"], "output_modalities": ["image"]},
         "supported_parameters": ["max_tokens"]},
    ]
}


def catalogue_http(payload=CATALOGUE, status: int = 200) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(status, json=payload)))


@pytest.fixture(autouse=True)
def _fresh_catalogue():
    llm_models._cache.clear()
    yield
    llm_models._cache.clear()


async def test_a_compatible_primary_and_different_fallback_validate():
    settings = make_settings(openrouter_model="google/gemini-2.5-flash-lite")
    result = await llm_models.validate_pair(catalogue_http(), settings, None, "openai/gpt-4o-mini")
    assert result == {"ok": True, "primaryModel": "google/gemini-2.5-flash-lite",
                      "fallbackModel": "openai/gpt-4o-mini", "problems": []}


@pytest.mark.parametrize(
    ("primary", "fallback", "problem"),
    [
        ("google/gemini-2.5-flash-lite", "google/gemini-2.5-flash-lite", "must be different"),
        ("google/gemini-2.5-flash-lite", "no/such-model", "not offered by OpenRouter"),
        ("google/gemini-2.5-flash-lite", "some/image-model", "does not produce text output"),
    ],
)
async def test_an_unusable_fallback_cannot_be_saved(primary, fallback, problem):
    result = await llm_models.validate_pair(catalogue_http(), make_settings(), primary, fallback)
    assert not result["ok"] and any(problem in p for p in result["problems"])


async def test_models_cannot_be_validated_when_the_catalogue_is_unreachable():
    result = await llm_models.validate_pair(catalogue_http(status=503), make_settings(), "a/b", "c/d")
    assert not result["ok"] and "could not be read" in result["problems"][0]


async def test_llm_options_offer_only_openrouter_and_only_compatible_models():
    options = await llm_models.llm_options(catalogue_http(), make_settings(llm_fallback_model="openai/gpt-4o-mini"))
    assert options["providers"] == ["openrouter"]
    assert options["defaultFallbackModel"] == "openai/gpt-4o-mini"
    assert [m["id"] for m in options["models"]["openrouter"]] == ["google/gemini-2.5-flash-lite", "openai/gpt-4o-mini"]
