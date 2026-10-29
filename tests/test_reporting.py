"""Reporting finished inbound calls to the dashboard: the event, its signature, retries and failure isolation."""

import asyncio
import hmac
import json
import time
import uuid
from datetime import datetime
from types import SimpleNamespace

import httpx
import pytest

from app import dispatch, reporting
from tests.test_api import ADMIN, APPOINTMENT_AGENT, client, converse  # noqa: F401
from tests.test_dispatch import SECRET, wait_until
from tests.test_inbound import NUMBER, another_reply, publish

URL = "https://dashboard.test/api/webhooks/voice-engine"


@pytest.fixture
def mailbox(client):  # noqa: F811
    """Stands in for the dashboard: records deliveries and answers with a scripted sequence of statuses."""
    box = SimpleNamespace(deliveries=[], attempts=0, statuses=[], delay=0.0)

    async def handler(request: httpx.Request) -> httpx.Response:
        box.attempts += 1
        if box.delay:
            await asyncio.sleep(box.delay)
        status = box.statuses.pop(0) if box.statuses else 200
        if status < 300:
            box.deliveries.append({"signature": request.headers["x-voice-engine-signature"], "body": request.content})
        return httpx.Response(status)

    services = client.app.state.services
    services.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    services.settings.dashboard_webhook_url = URL
    return box


def events(box) -> list[dict]:
    return [json.loads(d["body"]) for d in box.deliveries]


def finished_calls(client) -> list[dict]:  # noqa: F811
    return wait_until(lambda: [c for c in client.get("/calls", headers=ADMIN).json() if c["outcome"]] or None)


def outbox(client, where: str = "true"):  # noqa: F811
    pool = client.app.state.services.pool
    return client.portal.call(pool.fetch, f"SELECT * FROM webhook_outbox WHERE {where} ORDER BY id")


# ---- the event -------------------------------------------------------------------------------


def test_a_finished_inbound_call_is_reported_with_a_signed_event(client, mailbox):  # noqa: F811
    publish(client)
    converse(client, "CAin1", to_number=NUMBER)
    wait_until(lambda: mailbox.deliveries)
    time.sleep(0.3)  # and it is not sent twice

    assert len(mailbox.deliveries) == 1
    delivery = mailbox.deliveries[0]
    assert hmac.compare_digest(delivery["signature"], hmac.new(SECRET, delivery["body"], "sha256").hexdigest())

    event = events(mailbox)[0]
    call = finished_calls(client)[0]
    assert set(event) == {"schemaVersion", "eventId", "eventType", "occurredAt", "data"}  # no engineJobId
    assert (event["schemaVersion"], event["eventType"]) == (1, "call.completed")
    assert str(uuid.UUID(event["eventId"])) == event["eventId"]
    assert datetime.fromisoformat(event["occurredAt"]).tzinfo is not None
    assert event["data"] == {
        "direction": "inbound",
        "engineCallId": str(call["id"]),
        "dialledNumber": "918047491899",
        "callerNumber": "+919876543210",
        "durationSeconds": round(call["connected_seconds"]),
        "status": "completed",
        "outcome": "appointment_requested",
        "summary": "Caller wants an appointment.",
        "collectedFields": {"name": None, "preferred_time": "tomorrow 10am"},
        "transcriptRef": str(call["id"]),
    }


def test_the_dialled_number_is_reported_in_one_format_however_exotel_wrote_it(client, mailbox):  # noqa: F811
    publish(client)
    converse(client, "CAin1", to_number="08047491899")
    wait_until(lambda: mailbox.deliveries)
    assert events(mailbox)[0]["data"]["dialledNumber"] == "918047491899"


def test_a_call_answered_by_the_default_agent_is_reported_too(client, mailbox):  # noqa: F811
    client.post("/agents", json=APPOINTMENT_AGENT, headers=ADMIN)
    converse(client, "CAin1", to_number="+919000000000")
    wait_until(lambda: mailbox.deliveries)
    assert events(mailbox)[0]["data"]["dialledNumber"] == "919000000000"  # the dashboard decides whether it knows it


def test_an_outbound_call_is_not_reported_as_an_inbound_call(client, mailbox):  # noqa: F811
    agent = client.post("/agents", json=APPOINTMENT_AGENT, headers=ADMIN).json()
    client.app.state.services.settings.enable_outbound_calls = True
    client.post("/calls", json={"agent_id": agent["id"], "to": "+919876543210"}, headers=ADMIN)
    converse(client, "CAout1")
    finished_calls(client)
    time.sleep(0.4)
    assert mailbox.attempts == 0 and outbox(client) == []


def test_nothing_is_queued_when_reporting_is_not_configured(client, mailbox):  # noqa: F811
    client.app.state.services.settings.dashboard_webhook_url = None
    publish(client)
    converse(client, "CAin1", to_number=NUMBER)
    finished_calls(client)
    assert outbox(client) == [] and mailbox.attempts == 0

    client.app.state.services.settings.dashboard_webhook_url = URL
    client.app.state.services.settings.dashboard_webhook_secret = None  # a report could not be signed
    another_reply(client)
    converse(client, "CAin2", to_number=NUMBER)
    wait_until(lambda: len(finished_calls(client)) == 2)
    assert outbox(client) == []


def test_each_call_is_reported_once_even_if_asked_twice(client, mailbox):  # noqa: F811
    publish(client)
    converse(client, "CAin1", to_number=NUMBER)
    call = finished_calls(client)[0]
    wait_until(lambda: mailbox.deliveries)

    async def again(pool):
        async with pool.acquire() as conn:
            return await reporting.enqueue_inbound_call_report(conn, call["id"], URL)

    assert client.portal.call(again, client.app.state.services.pool) is False
    assert len(outbox(client, f"call_id = {call['id']}")) == 1


# ---- delivery --------------------------------------------------------------------------------


def test_a_failed_delivery_is_retried_and_arrives_once(client, mailbox, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dispatch, "retry_delay", lambda attempts: 0.05)
    mailbox.statuses = [500, 503]
    publish(client)
    converse(client, "CAin1", to_number=NUMBER)

    wait_until(lambda: mailbox.deliveries)
    time.sleep(0.3)
    assert (mailbox.attempts, len(mailbox.deliveries)) == (3, 1)
    assert outbox(client)[0]["delivered_at"] is not None


def test_a_timeout_or_rate_limit_is_retried_but_a_refusal_is_final(client, mailbox, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dispatch, "retry_delay", lambda attempts: 0.05)
    mailbox.statuses = [429, 408]
    publish(client)
    converse(client, "CAin1", to_number=NUMBER)
    wait_until(lambda: mailbox.deliveries)
    assert mailbox.attempts == 3

    mailbox.statuses = [404]  # the dashboard does not know this number: retrying cannot help
    attempts_before = mailbox.attempts
    another_reply(client)
    converse(client, "CAin2", to_number=NUMBER)
    wait_until(lambda: outbox(client, "abandoned_at IS NOT NULL"))
    time.sleep(0.4)
    abandoned = outbox(client, "abandoned_at IS NOT NULL")
    assert len(abandoned) == 1 and abandoned[0]["last_error"] == "HTTP 404" and abandoned[0]["delivered_at"] is None
    assert mailbox.attempts == attempts_before + 1


def test_retries_are_bounded_and_the_report_is_kept(client, mailbox, monkeypatch):  # noqa: F811
    monkeypatch.setattr(dispatch, "retry_delay", lambda attempts: 0.02)
    client.app.state.services.settings.webhook_max_attempts = 3
    mailbox.statuses = [500] * 50
    publish(client)
    converse(client, "CAin1", to_number=NUMBER)

    wait_until(lambda: outbox(client, "abandoned_at IS NOT NULL"))
    time.sleep(0.4)
    assert mailbox.attempts == 3 and mailbox.deliveries == []
    row = outbox(client)[0]
    assert (row["attempts"], row["last_error"]) == (3, "HTTP 500")
    assert row["payload"]["data"]["outcome"] == "appointment_requested"  # kept for inspection


def test_a_slow_dashboard_does_not_hold_up_calls(client, mailbox):  # noqa: F811
    mailbox.delay = 1.5
    publish(client)
    converse(client, "CAin1", to_number=NUMBER)
    wait_until(lambda: mailbox.attempts == 1)  # the first report is now stuck waiting for the dashboard

    another_reply(client)
    started = time.monotonic()
    converse(client, "CAin2", to_number=NUMBER)
    assert len(finished_calls(client)) >= 1
    wait_until(lambda: len([c for c in client.get("/calls", headers=ADMIN).json() if c["outcome"]]) == 2)
    assert time.monotonic() - started < 1.2
    assert mailbox.deliveries == []  # still waiting on the dashboard, while the second call completed and was saved
    wait_until(lambda: len(mailbox.deliveries) == 2, timeout=10)


def test_a_report_that_cannot_be_queued_never_loses_the_call(client, mailbox, monkeypatch):  # noqa: F811
    async def broken(*_args, **_kwargs):
        raise RuntimeError("outbox unavailable")

    monkeypatch.setattr(reporting, "enqueue_inbound_call_report", broken)
    publish(client)
    converse(client, "CAin1", to_number=NUMBER)

    call = finished_calls(client)[0]
    assert call["status"] == "completed" and call["outcome"] == "appointment_requested"
    assert mailbox.attempts == 0 and outbox(client) == []


# ---- the payload as a pure function -----------------------------------------------------------


def row(**over):
    base = {
        "id": 7,
        "to_number": "08047491899",
        "from_number": " +919876543210 ",
        "status": "completed",
        "connected_seconds": 41.6,
        "outcome": "callback_requested",
        "summary": "Wants a call tomorrow.",
        "outcome_data": {"name": "Ravi"},
    }
    return base | over


def test_the_event_for_an_empty_or_odd_call_stays_valid():
    event = reporting.inbound_call_event(
        row(to_number="", from_number=None, connected_seconds=None, summary=None, outcome_data=None),
        has_transcript=False,
    )
    assert event["data"] == {
        "direction": "inbound",
        "engineCallId": "7",
        "dialledNumber": "",
        "callerNumber": None,
        "durationSeconds": 0,
        "status": "completed",
        "outcome": "callback_requested",
        "summary": "",
        "collectedFields": {},
        "transcriptRef": None,
    }


def test_the_event_rounds_the_duration_trims_numbers_and_limits_the_summary():
    event = reporting.inbound_call_event(row(summary="x" * 10_000), has_transcript=True)
    data = event["data"]
    assert (data["durationSeconds"], data["dialledNumber"], data["callerNumber"]) == (
        42,
        "918047491899",
        "+919876543210",
    )
    assert len(data["summary"]) == reporting.SUMMARY_LIMIT and data["transcriptRef"] == "7"
