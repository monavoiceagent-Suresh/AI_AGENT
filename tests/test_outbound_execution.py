"""Focused outbound contracts, using the existing queue and fake persistence/telephony."""

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import asyncpg
import pytest
from pydantic import ValidationError

from app import api, calls, db, dispatch, dispatch_db
from app.models import DispatchIn, DispatchOut, TestCallIn as OutboundTestRequest
from app.providers.exotel import ExotelError
from tests.test_outbound_dial import AGENT, FakePool, services

pytestmark = pytest.mark.anyio

PERSONA = {
    "key": "sales", "name": "Sales", "language": "en", "workflow": "general",
    "instructions": "Saved prompt", "greeting": "Hello", "knowledge": "", "version": 7,
}


def request(**overrides):
    return DispatchIn(**({
        "idempotencyKey": "request-key", "campaignId": "campaign", "clientId": "tenant",
        "persona": PERSONA, "callingWindow": {"start": "09:00", "end": "18:00"},
        "retryPolicy": {"maxTries": 3, "retryGapMinutes": 30, "callbackWindowDays": 3},
        "leads": [{"id": "lead", "phone": "+919876543210", "name": "Asha"}],
        "webhookUrl": "https://mona.example/api/webhooks/voice-engine",
    } | overrides))


async def test_reported_test_call_uses_the_existing_dispatcher_with_one_attempt(monkeypatch):
    accept = AsyncMock(return_value=DispatchOut(engine_job_id="engine-test", status="accepted"))
    monkeypatch.setattr(dispatch, "accept", accept)
    body = OutboundTestRequest(
        client_id="tenant", persona=PERSONA, to="+919876543210", contact_name="Asha",
        idempotency_key="request-key", webhook_url="https://mona.example/hook",
        voice_id="voice", tts_model_id="m1",
    )
    result = await api.place_test_call(body, SimpleNamespace())
    assert result.status == "accepted"
    queued = accept.call_args.args[1]
    assert queued.client_id == "tenant" and queued.persona.instructions == "Saved prompt"
    assert queued.retry_policy.max_tries == 1 and queued.campaign_id == "test-request-key"
    assert queued.voice_id == "voice" and queued.tts_model_id == "m1"
    assert queued.leads[0].phone == body.to


def test_schedule_requires_an_explicit_timezone():
    with pytest.raises(ValidationError, match="timezone"):
        request(notBefore="2027-01-04T10:30:00")
    assert request(notBefore="2027-01-04T10:30:00+05:30").not_before.astimezone(UTC) == datetime(
        2027, 1, 4, 5, tzinfo=UTC
    )


class Connection:
    def __init__(self):
        self.leads = []
        self.events = []

    @asynccontextmanager
    async def transaction(self):
        yield

    async def fetchval(self, sql, *args):
        return "engine-job" if "dispatch_jobs" in sql else 3

    async def executemany(self, sql, rows):
        self.leads = list(rows)


class Pool:
    def __init__(self, connection):
        self.connection = connection

    @asynccontextmanager
    async def acquire(self):
        yield self.connection


async def test_schedule_is_persisted_as_existing_lead_due_time():
    conn = Connection()
    future = datetime(2027, 1, 4, 5, tzinfo=UTC)
    await dispatch_db.create_job(Pool(conn), request(notBefore=future))
    assert conn.leads == [("engine-job", "lead", "+919876543210", "Asha", future, 0)]


async def test_resumed_lead_keeps_attempts_and_the_later_retry_due_time():
    conn = Connection()
    future = datetime(2027, 1, 4, 5, tzinfo=UTC)
    retry = datetime(2027, 1, 4, 6, tzinfo=UTC)
    await dispatch_db.create_job(Pool(conn), request(notBefore=future, leads=[{
        "id": "lead", "phone": "+919876543210", "attempts": 2, "nextAttemptAt": retry,
    }]))
    assert conn.leads[0][-2:] == (retry, 2)


async def test_concurrent_job_insert_conflict_returns_the_same_job(monkeypatch):
    class Conflict(Connection):
        async def fetchval(self, sql, *args):
            if "INSERT INTO dispatch_jobs" in sql:
                error = asyncpg.UniqueViolationError("duplicate idempotency key")
                error.constraint_name = "dispatch_jobs_idempotency_key_key"
                raise error
            return 3

    monkeypatch.setattr(dispatch_db, "job_by_key", AsyncMock(return_value={"id": "already-accepted"}))
    assert await dispatch_db.create_job(Pool(Conflict()), request()) == "already-accepted"


async def test_dispatch_emits_started_with_existing_tenant_job_and_caller_id(monkeypatch):
    lead = {
        "lead_pk": 1, "lead_id": "lead", "phone": "+919876543210", "name": "Asha",
        "attempts": 0, "agent_id": 3, "job_id": "engine-job", "voice_id": "voice", "tts_model_id": "m1",
    }
    svc = services(FakePool())
    monkeypatch.setattr(dispatch_db, "next_due_lead", AsyncMock(return_value=lead))
    monkeypatch.setattr(dispatch_db, "claim_lead", AsyncMock(return_value=True))
    monkeypatch.setattr(db, "get_agent", AsyncMock(return_value={}))
    monkeypatch.setattr(dispatch, "Agent", lambda **_: AGENT)
    placed = AsyncMock(return_value=42)
    monkeypatch.setattr(calls, "place_outbound_call", placed)
    monkeypatch.setattr(dispatch_db, "mark_job_running", AsyncMock())
    enqueue = AsyncMock()
    monkeypatch.setattr(dispatch_db, "enqueue_event", enqueue)
    assert await dispatch.place_next_call(svc)
    assert placed.call_args.kwargs == {"voice_id": "voice", "tts_model_id": "m1"}
    event = enqueue.call_args.args[2]
    assert event["engineJobId"] == "engine-job" and event["eventType"] == "call.started"
    assert event["data"]["engineCallId"] == "42"
    assert event["data"]["callerNumber"] == svc.settings.exotel_caller_id


async def test_concurrent_outbound_requests_cannot_take_the_same_capacity_slot():
    svc = services(FakePool())
    svc.settings.max_concurrent_calls = 1
    entered, release = asyncio.Event(), asyncio.Event()

    async def dial(*args, **kwargs):
        entered.set()
        await release.wait()
        return "CAone"

    svc.exotel.connect_to_flow = dial
    first = asyncio.create_task(calls.place_outbound_call(svc, AGENT, "+919876543210", None))
    await entered.wait()
    try:
        with pytest.raises(calls.CallRejected, match="limit"):
            await calls.place_outbound_call(svc, AGENT, "+919876543211", None)
    finally:
        release.set()
        await first
    assert list(svc.ringing) == ["CAone"]


async def test_failed_placement_releases_capacity():
    svc = services(FakePool())
    svc.exotel.connect_to_flow = AsyncMock(side_effect=ExotelError("provider refused"))
    with pytest.raises(ExotelError):
        await calls.place_outbound_call(svc, AGENT, "+919876543210", None)
    assert not svc.ringing


async def test_nonterminal_callbacks_do_not_end_a_ringing_call(monkeypatch):
    svc = services(FakePool())
    svc.ringing["CAone"] = 0
    update = AsyncMock(return_value=True)
    monkeypatch.setattr(db, "apply_status_callback", update)
    await calls.call_status(svc, "CAone", "ringing")
    assert "CAone" in svc.ringing
    update.assert_not_called()
    await calls.call_status(svc, "CAone", "no-answer")
    update.assert_awaited_once_with(svc.pool, "CAone", "no_answer")
    assert "CAone" not in svc.ringing


@pytest.mark.parametrize("unknown, state", [(False, "retry"), (True, "exhausted")])
async def test_only_confirmed_dial_failures_are_automatically_retried(monkeypatch, unknown, state):
    svc = services(FakePool())
    lead = {
        "lead_pk": 1, "lead_id": "lead", "phone": "+919876543210", "name": "Asha",
        "attempts": 0, "agent_id": 3, "job_id": "job", "job_status": "running",
        "max_tries": 3, "retry_gap_minutes": 30, "voice_id": None, "tts_model_id": None,
    }
    monkeypatch.setattr(dispatch_db, "next_due_lead", AsyncMock(return_value=lead))
    monkeypatch.setattr(dispatch_db, "claim_lead", AsyncMock(return_value=True))
    monkeypatch.setattr(db, "get_agent", AsyncMock(return_value={}))
    monkeypatch.setattr(dispatch, "Agent", lambda **_: AGENT)
    monkeypatch.setattr(calls, "place_outbound_call", AsyncMock(side_effect=ExotelError("failed", placement_unknown=unknown)))
    settle = AsyncMock()
    monkeypatch.setattr(dispatch_db, "settle_lead", settle)
    assert await dispatch.place_next_call(svc)
    event = settle.call_args.kwargs["event"]
    assert event["eventType"] == "call.failed" and event["data"]["leadState"] == state
    assert ("nextAttemptAt" in event["data"]) is not unknown


async def test_stale_claim_recovery_reports_unknown_placement_instead_of_redialling(monkeypatch):
    conn = Connection()
    conn.fetch = AsyncMock(return_value=[{"job_id": "job", "lead_id": "lead", "phone": "+919876543210", "attempts": 1}])
    enqueue = AsyncMock()
    monkeypatch.setattr(dispatch_db, "enqueue_event", enqueue)
    assert await dispatch_db.release_unplaced_claims(Pool(conn), 300) == 1
    event = enqueue.call_args.args[2]
    assert event["eventType"] == "call.failed"
    assert event["data"]["outcome"] == "placement_unknown"
    assert event["data"]["leadState"] == "exhausted"


async def test_health_advertises_the_reported_outbound_contract():
    svc = SimpleNamespace(pool=SimpleNamespace(fetchval=AsyncMock(return_value=1)), active_calls=0)
    assert (await api.health(svc))["outbound_dispatch_version"] == 1
