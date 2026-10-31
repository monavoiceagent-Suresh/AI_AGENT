"""Campaign dispatch from the dashboard: contract, calling, retries, cancellation, signed webhooks."""

import hmac
import json
import time
import uuid
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest

from app import dispatch
from tests.test_api import ADMIN, APPOINTMENT_AGENT, client, converse  # noqa: F401

SECRET = b"test-dashboard-secret"
IST = timezone(timedelta(hours=5, minutes=30))
PERSONA = {
    "key": "october-14",
    "name": "October follow-ups",
    "language": "en",
    "workflow": "appointment",
    "instructions": "You book appointments for Sunrise Clinic. Context: [[CALLER_CONTEXT]]",
    "greeting": "Hello from Sunrise Clinic.",
    "knowledge": "",
    "version": 1,
}


def dispatch_body(**overrides) -> dict:
    body = {
        "idempotencyKey": uuid.uuid4().hex,
        "campaignId": "campaign-1",
        "clientId": "client-1",
        "persona": PERSONA,
        "callingWindow": {"start": "00:00", "end": "23:59"},
        "retryPolicy": {"maxTries": 2, "retryGapMinutes": 1, "callbackWindowDays": 3},
        "leads": [{"id": "lead-1", "phone": "+919876543210", "name": "Ravi"}],
        "webhookUrl": "https://dashboard.test/api/webhooks/voice-engine",
    }
    return body | overrides


def wait_until(predicate, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if value := predicate():
            return value
        time.sleep(0.05)
    raise AssertionError("condition not met in time")


@pytest.fixture
def sink(client):  # noqa: F811
    """Replaces the engine's HTTP client so webhook deliveries are recorded instead of sent."""
    deliveries: list[dict] = []
    failures = SimpleNamespace(remaining=0)

    def handler(request: httpx.Request) -> httpx.Response:
        if failures.remaining:
            failures.remaining -= 1
            return httpx.Response(500)
        deliveries.append({"signature": request.headers["x-voice-engine-signature"], "body": request.content})
        return httpx.Response(200)

    client.app.state.services.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return SimpleNamespace(
        deliveries=deliveries, failures=failures, types=lambda: [event(d)["eventType"] for d in deliveries]
    )


def event(delivery: dict) -> dict:
    return json.loads(delivery["body"])


@pytest.fixture
def ready(client, sink):  # noqa: F811
    """Outbound calling switched on and webhook deliveries captured."""
    client.app.state.services.settings.enable_outbound_calls = True
    return client


def start(client, **overrides) -> str:  # noqa: F811
    response = client.post("/campaigns/dispatch", json=dispatch_body(**overrides), headers=ADMIN)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "accepted", response.text
    return response.json()["engineJobId"]


def first_call_id(client) -> int:  # noqa: F811
    calls = wait_until(lambda: client.get("/calls", headers=ADMIN).json())
    return calls[0]["id"]


def no_answer(client, call_sid: str) -> None:  # noqa: F811
    assert (
        client.post("/exotel/status?token=test-hook", data={"CallSid": call_sid, "Status": "no-answer"}).status_code
        == 200
    )


def run_now(client, sql: str) -> None:  # noqa: F811
    client.portal.call(client.app.state.services.pool.execute, sql)


# ---- pure decisions --------------------------------------------------------------------------

NOW = datetime(2026, 10, 27, 10, 0, tzinfo=UTC)


def attempt(**overrides) -> dict:
    return {"attempts": 1, "max_tries": 3, "retry_gap_minutes": 30, "job_status": "running"} | overrides


@pytest.mark.parametrize(
    ("overrides", "connected", "outcome", "expected"),
    [
        ({}, True, "appointment_requested", ("done", "done", None)),
        ({}, True, "no_conversation", ("pending", "retry", NOW + timedelta(minutes=30))),
        ({}, False, "no_answer", ("pending", "retry", NOW + timedelta(minutes=30))),
        ({"attempts": 3}, False, "busy", ("failed", "exhausted", None)),
        ({"attempts": 3}, True, "no_conversation", ("failed", "exhausted", None)),
        ({"job_status": "cancelled"}, False, "no_answer", ("pending", "queued", None)),
        ({"job_status": "cancelled"}, True, "interested", ("done", "done", None)),
    ],
)
def test_decide(overrides, connected, outcome, expected):
    assert dispatch.decide(attempt(**overrides), connected=connected, outcome=outcome, now=NOW) == expected


def test_retry_delay_backs_off_and_is_capped():
    assert [dispatch.retry_delay(n) for n in (1, 2, 3)] == [5.0, 10.0, 20.0]
    assert dispatch.retry_delay(50) == 900.0


# ---- the contract ----------------------------------------------------------------------------


def test_dispatch_needs_the_token_and_a_valid_body(client):  # noqa: F811
    assert client.post("/campaigns/dispatch", json=dispatch_body()).status_code == 401
    bad_phone = dispatch_body(leads=[{"id": "l1", "phone": "12345", "name": "x"}])
    assert client.post("/campaigns/dispatch", json=bad_phone, headers=ADMIN).status_code == 422
    backwards = dispatch_body(callingWindow={"start": "18:00", "end": "09:00"})
    assert client.post("/campaigns/dispatch", json=backwards, headers=ADMIN).status_code == 422
    no_leads = dispatch_body(leads=[])
    assert client.post("/campaigns/dispatch", json=no_leads, headers=ADMIN).status_code == 422
    no_persona = dispatch_body()
    del no_persona["persona"]
    assert client.post("/campaigns/dispatch", json=no_persona, headers=ADMIN).status_code == 422
    empty_greeting = dispatch_body(persona=PERSONA | {"greeting": ""})
    assert client.post("/campaigns/dispatch", json=empty_greeting, headers=ADMIN).status_code == 422


def shared_agent(client) -> dict:  # noqa: F811
    return next(a for a in client.get("/agents", headers=ADMIN).json() if a["persona_key"] == PERSONA["key"])


def test_a_campaign_voice_reaches_the_tts_request_and_leaves_the_shared_agent_alone(ready):
    start(ready, voiceId="campaign-voice", ttsModelId="m1")
    wait_until(lambda: ready.app.state.services.exotel.dialled)
    first_call_id(ready)
    converse(ready, "CAout1")

    assert set(ready.app.state.services.tts.voices) == {("campaign-voice", "m1")}
    assert shared_agent(ready)["voice_id"] is None


def test_two_campaigns_on_one_persona_each_speak_with_their_own_voice(ready):
    start(ready, voiceId="voice-one", ttsModelId="m1")
    start(
        ready,
        campaignId="campaign-2",
        voiceId="voice-two",
        ttsModelId="standard",
        leads=[{"id": "lead-2", "phone": "+919876543211"}],
    )
    services = ready.app.state.services
    services.llm.replies.append("Thank you, goodbye. <end_call>")

    wait_until(lambda: len(services.exotel.dialled) == 1)
    wait_until(lambda: len(ready.get("/calls", headers=ADMIN).json()) == 1)
    converse(ready, "CAout1")
    first_call_voices = len(services.tts.voices)
    wait_until(lambda: len(services.exotel.dialled) == 2)
    wait_until(lambda: len(ready.get("/calls", headers=ADMIN).json()) == 2)
    converse(ready, "CAout2")

    assert set(services.tts.voices[:first_call_voices]) == {("voice-one", "m1")}
    assert set(services.tts.voices[first_call_voices:]) == {("voice-two", "standard")}
    assert shared_agent(ready)["voice_id"] is None


def test_a_campaign_without_a_voice_speaks_the_language_default(ready):
    start(ready)
    wait_until(lambda: ready.app.state.services.exotel.dialled)
    first_call_id(ready)
    converse(ready, "CAout1")

    assert set(ready.app.state.services.tts.voices) == {("en-voice", "m1")}


def test_each_call_end_is_logged_with_its_campaign_persona_voice_and_outcome(ready, caplog):
    caplog.set_level("INFO")
    start(ready, voiceId="campaign-voice", ttsModelId="m1")
    wait_until(lambda: ready.app.state.services.exotel.dialled)
    first_call_id(ready)
    converse(ready, "CAout1")
    wait_until(lambda: any("ended: campaign=" in r.getMessage() for r in caplog.records))

    line = next(r.getMessage() for r in caplog.records if "ended: campaign=" in r.getMessage())
    assert "campaign=campaign-1" in line and f"persona={PERSONA['key']}" in line
    assert "voice=campaign-voice/m1" in line and "duration=" in line and "outcome=" in line and "cost_inr=" in line
    assert "+919876543210" not in line and "Ravi" not in line


def test_a_dispatched_campaigns_outcome_rules_are_applied_to_its_agent(ready):
    persona = PERSONA | {
        "outcomeRules": [{"name": "Interested", "definition": "Clear positive response."}],
        "outcomeFallback": "No decision.",
    }
    start(ready, persona=persona)

    agents = ready.get("/agents", headers=ADMIN).json()
    agent = next(a for a in agents if a["persona_key"] == PERSONA["key"])
    assert agent["outcome_rules"] == [{"name": "Interested", "definition": "Clear positive response."}]
    assert agent["outcome_fallback"] == "No decision."


def test_dispatch_rejects_an_outcome_rule_named_inconclusive(ready):
    persona = PERSONA | {"outcomeRules": [{"name": "Inconclusive", "definition": "x"}], "outcomeFallback": "y"}
    body = dispatch_body(persona=persona)
    rejected = ready.post("/campaigns/dispatch", json=body, headers=ADMIN)
    assert rejected.status_code == 422


def test_dispatch_rejects_a_half_set_voice_choice(ready):
    body = dispatch_body(voiceId="only-this")
    rejected = ready.post("/campaigns/dispatch", json=body, headers=ADMIN).json()
    assert rejected["status"] == "rejected" and "together" in rejected["reason"]


def test_dispatch_is_rejected_with_a_reason_until_calling_is_possible(client, sink):  # noqa: F811
    body = dispatch_body()
    outbound_off = client.post("/campaigns/dispatch", json=body, headers=ADMIN).json()
    assert outbound_off["status"] == "rejected" and "ENABLE_OUTBOUND_CALLS" in outbound_off["reason"]

    settings = client.app.state.services.settings
    settings.enable_outbound_calls = True
    settings.bakbak_voices.pop("te")
    telugu = client.post("/campaigns/dispatch", json=dispatch_body(persona=PERSONA | {"language": "te"}), headers=ADMIN)
    assert telugu.json()["status"] == "rejected" and "BAKBAK_VOICES" in telugu.json()["reason"]

    settings.dashboard_webhook_secret = None
    no_secret = client.post("/campaigns/dispatch", json=body, headers=ADMIN).json()
    assert no_secret["status"] == "rejected" and "DASHBOARD_WEBHOOK_SECRET" in no_secret["reason"]


def test_each_client_gets_its_own_agent_for_a_persona_key(ready):
    start(ready)
    start(ready, clientId="client-2", campaignId="campaign-2", persona=PERSONA | {"name": "Other client's"})
    agents = ready.get("/agents", headers=ADMIN).json()
    assert sorted((a["client_id"], a["persona_key"], a["name"]) for a in agents) == [
        ("client-1", "october-14", "October follow-ups"),
        ("client-2", "october-14", "Other client's"),
    ]

    taken = APPOINTMENT_AGENT | {"is_inbound_default": False, "client_id": "client-1", "persona_key": "october-14"}
    assert ready.post("/agents", json=taken, headers=ADMIN).status_code == 409
    assert ready.post("/agents", json=taken | {"client_id": None}, headers=ADMIN).status_code == 422


def test_a_newer_persona_version_refreshes_the_agent_but_keeps_its_voice_settings(ready):
    start(ready)
    agent = ready.get("/agents", headers=ADMIN).json()[0]
    custom = {**agent, "voice_id": "custom-voice", "tts_model_id": "m1"}
    assert ready.put(f"/agents/{agent['id']}", json=custom, headers=ADMIN).status_code == 200

    newer = PERSONA | {"instructions": "New text. Context: [[CALLER_CONTEXT]]", "version": 2}
    run_now(ready, "UPDATE dispatch_jobs SET finalized_at = now()")  # lets the campaign be started again
    start(ready, persona=newer)

    refreshed = ready.get("/agents", headers=ADMIN).json()
    assert len(refreshed) == 1
    assert refreshed[0]["instructions"].startswith("New text.")
    assert (refreshed[0]["voice_id"], refreshed[0]["tts_model_id"]) == ("custom-voice", "m1")

    other_language = PERSONA | {"language": "hi", "version": 3}
    run_now(ready, "UPDATE dispatch_jobs SET finalized_at = now()")
    start(ready, persona=other_language)
    switched = ready.get("/agents", headers=ADMIN).json()[0]
    assert (switched["language"], switched["voice_id"], switched["tts_model_id"]) == ("hi", None, None)


def test_dispatch_is_idempotent_and_allows_one_open_job_per_campaign(ready):
    body = dispatch_body()
    first = ready.post("/campaigns/dispatch", json=body, headers=ADMIN).json()
    replay = ready.post("/campaigns/dispatch", json=body, headers=ADMIN).json()
    assert first["status"] == replay["status"] == "accepted" and first["engineJobId"] == replay["engineJobId"]

    second = ready.post("/campaigns/dispatch", json=dispatch_body(), headers=ADMIN).json()
    assert second["status"] == "rejected" and "unfinished dispatch" in second["reason"]
    assert len(ready.app.state.services.exotel.dialled) <= 1  # the replay never dials twice


def test_unknown_job_is_not_found(ready):
    missing = uuid.uuid4()
    assert ready.get(f"/jobs/{missing}", headers=ADMIN).status_code == 404
    assert ready.post(f"/jobs/{missing}/cancel", headers=ADMIN).status_code == 404


# ---- calling and reporting -------------------------------------------------------------------


def test_the_call_speaks_the_persona_it_was_started_with(ready):
    persona = PERSONA | {
        "instructions": "You are Zed from Acme Dental. Facts: [[KNOWLEDGE]] Context: [[CALLER_CONTEXT]]",
        "greeting": "Namaste, this is Zed from Acme Dental.",
        "knowledge": "Cleanings cost nothing this month.",
    }
    start(ready, persona=persona)
    wait_until(lambda: ready.app.state.services.exotel.dialled)
    first_call_id(ready)
    converse(ready, "CAout1")

    services = ready.app.state.services
    system_prompt = services.llm.requests[0][0]["content"]
    assert "You are Zed from Acme Dental." in system_prompt
    assert "Cleanings cost nothing this month." in system_prompt
    assert "Outbound call to Ravi." in system_prompt and "[[" not in system_prompt
    assert services.tts.texts[0] == "Namaste, this is Zed from Acme Dental."


def test_each_campaign_keeps_the_persona_version_it_was_started_with(ready):
    version_one = PERSONA | {"instructions": "VERSION ONE text. Context: [[CALLER_CONTEXT]]", "version": 1}
    version_two = PERSONA | {"instructions": "VERSION TWO text. Context: [[CALLER_CONTEXT]]", "version": 2}
    start(ready, persona=version_one)
    start(ready, campaignId="campaign-2", persona=version_two, leads=[{"id": "lead-2", "phone": "+919876543211"}])
    services = ready.app.state.services
    services.llm.replies.append("Thank you, goodbye. <end_call>")

    wait_until(lambda: len(services.exotel.dialled) == 1)
    wait_until(lambda: len(ready.get("/calls", headers=ADMIN).json()) == 1)
    converse(ready, "CAout1")
    wait_until(lambda: len(services.exotel.dialled) == 2)  # capacity frees once the first call is over
    wait_until(lambda: len(ready.get("/calls", headers=ADMIN).json()) == 2)
    converse(ready, "CAout2")

    first, second = (request[0]["content"] for request in services.llm.requests)
    assert "VERSION ONE" in first and "VERSION TWO" not in first
    assert "VERSION TWO" in second and "VERSION ONE" not in second
    assert [a["instructions"][:11] for a in ready.get("/agents", headers=ADMIN).json()] == ["VERSION TWO"]


def test_a_dispatched_lead_is_called_and_reported_with_a_signed_webhook(ready, sink):
    job_id = start(ready)
    wait_until(lambda: ready.app.state.services.exotel.dialled)
    call_id = first_call_id(ready)
    converse(ready, "CAout1")

    wait_until(lambda: "job.completed" in sink.types())
    assert sink.types() == ["call.completed", "job.completed"]
    for delivery in sink.deliveries:
        expected = hmac.new(SECRET, delivery["body"], "sha256").hexdigest()
        assert hmac.compare_digest(delivery["signature"], expected)

    call_event, job_event = (event(d) for d in sink.deliveries)
    assert call_event["engineJobId"] == job_event["engineJobId"] == job_id
    assert call_event["data"] == {
        "leadId": "lead-1",
        "leadPhone": "+919876543210",
        "outcome": "appointment_requested",
        "durationSeconds": call_event["data"]["durationSeconds"],
        "attempt": 1,
        "leadState": "done",
        "engineCallId": str(call_id),
        "transcriptRef": str(call_id),
    }
    assert job_event["data"]["minutesUsed"] >= 0 and job_event["data"]["leadsDone"] == 1

    job = ready.get(f"/jobs/{job_id}", headers=ADMIN).json()
    assert job == {"engineJobId": job_id, "status": "completed", "callsCompleted": 1, "callsFailed": 0}
    assert ready.get(f"/calls/{call_id}", headers=ADMIN).json()["contact_name"] == "Ravi"


def test_an_unanswered_lead_is_retried_and_then_given_up_on(ready, sink):
    job_id = start(ready)
    wait_until(lambda: ready.app.state.services.exotel.dialled)
    first_call_id(ready)
    no_answer(ready, "CAout1")

    wait_until(lambda: sink.types() == ["call.failed"])
    retry = event(sink.deliveries[0])["data"]
    assert retry["outcome"] == "no_answer" and retry["leadState"] == "retry" and retry["attempt"] == 1
    assert datetime.fromisoformat(retry["nextAttemptAt"]) > datetime.now(UTC)
    assert len(ready.app.state.services.exotel.dialled) == 1  # the gap has not passed

    run_now(ready, "UPDATE dispatch_leads SET next_attempt_at = now()")
    wait_until(lambda: len(ready.app.state.services.exotel.dialled) == 2)
    wait_until(lambda: len(ready.get("/calls", headers=ADMIN).json()) == 2)
    no_answer(ready, "CAout2")

    wait_until(lambda: "job.completed" in sink.types())
    last = event(sink.deliveries[1])["data"]
    assert last["leadState"] == "exhausted" and last["attempt"] == 2
    job = ready.get(f"/jobs/{job_id}", headers=ADMIN).json()
    assert (job["status"], job["callsCompleted"], job["callsFailed"]) == ("completed", 0, 1)


def test_cancelling_stops_new_calls_and_still_reports_the_one_in_flight(ready, sink):
    leads = [{"id": f"lead-{n}", "phone": f"+91987654321{n}", "name": "A"} for n in range(2)]
    job_id = start(ready, leads=leads)
    wait_until(lambda: ready.app.state.services.exotel.dialled)
    first_call_id(ready)

    assert ready.post(f"/jobs/{job_id}/cancel", headers=ADMIN).json() == {"status": "cancelled"}
    assert ready.post(f"/jobs/{job_id}/cancel", headers=ADMIN).json() == {"status": "cancelled"}
    no_answer(ready, "CAout1")

    wait_until(lambda: "job.cancelled" in sink.types())
    assert event(sink.deliveries[0])["data"]["leadState"] == "queued"
    assert len(ready.app.state.services.exotel.dialled) == 1

    # The campaign can be dispatched again once the cancelled job has finished.
    assert start(ready)


def test_nothing_is_dialled_outside_the_calling_window(ready, sink):
    hour = datetime.now(IST).hour
    window = {"start": "13:00", "end": "14:00"} if hour < 12 else {"start": "01:00", "end": "02:00"}
    start(ready, callingWindow=window)
    time.sleep(0.5)
    assert ready.app.state.services.exotel.dialled == []
    assert sink.deliveries == []


def test_a_failed_delivery_is_retried_without_losing_or_reordering_events(ready, sink, monkeypatch):
    monkeypatch.setattr(dispatch, "retry_delay", lambda attempts: 0.05)
    sink.failures.remaining = 2
    start(ready, retryPolicy={"maxTries": 1, "retryGapMinutes": 1, "callbackWindowDays": 3})
    wait_until(lambda: ready.app.state.services.exotel.dialled)
    first_call_id(ready)
    no_answer(ready, "CAout1")

    wait_until(lambda: "job.completed" in sink.types())
    assert sink.types() == ["call.failed", "job.completed"]
    assert event(sink.deliveries[0])["data"]["leadState"] == "exhausted"
