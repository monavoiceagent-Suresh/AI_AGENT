"""Campaign dispatch: accepts jobs from the dashboard, places their calls, records every attempt and
reports results back by signed webhook.

One engine process owns the queue (calls are placed by an in-process loop), so run a single instance.
"""

import asyncio
import hashlib
import hmac
import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from app import calls, db, dispatch_db
from app.calls import CallRejected, Services
from app.models import Agent, AgentIn, DispatchIn, DispatchOut
from app.providers.exotel import ExotelError

log = logging.getLogger(__name__)

# A ringing call normally ends within Exotel's ring timeout and reports through the status callback.
# These bound how long a call that never reported back can hold its lead.
RINGING_GRACE_SECONDS = 300
CONNECTED_GRACE_SECONDS = 300
UNPLACED_CLAIM_SECONDS = 300
WEBHOOK_TIMEOUT_SECONDS = 10
MAX_DELIVERY_ROUNDS = 20
MAX_PARALLEL_DELIVERIES = 8
# A 4xx other than these is the dashboard saying it will never accept this event.
RETRYABLE_CLIENT_ERRORS = frozenset({408, 425, 429})
# A connected call whose outcome means nobody was really reached is attempted again.
RETRY_OUTCOMES = frozenset({"no_conversation"})


def _rejected(reason: str) -> DispatchOut:
    return DispatchOut(engine_job_id=None, status="rejected", reason=reason)


async def accept(svc: Services, request: DispatchIn) -> DispatchOut:
    existing = await dispatch_db.job_by_key(svc.pool, request.idempotency_key)
    if existing is not None:
        return DispatchOut(engine_job_id=str(existing["id"]), status="accepted")
    if svc.settings.dashboard_webhook_secret is None:
        return _rejected("DASHBOARD_WEBHOOK_SECRET is not configured on the voice engine")
    if not svc.settings.enable_outbound_calls:
        return _rejected("Outbound calls are disabled on the voice engine (ENABLE_OUTBOUND_CALLS=false)")
    persona = request.persona
    speakable = AgentIn(
        **persona.model_dump(include=set(dispatch_db.PERSONA_FIELDS)),
        voice_id=request.voice_id,
        tts_model_id=request.tts_model_id,
    )
    if problem := calls.validate_agent(speakable, svc.settings):
        return _rejected(f"Persona {persona.key!r} cannot be spoken by the voice engine: {problem}")
    try:
        job_id = await dispatch_db.create_job(svc.pool, request)
    except dispatch_db.OpenJobExists:
        return _rejected("This campaign already has an unfinished dispatch on the voice engine")
    log.info(
        "dispatch job %s accepted: %d leads, persona %s v%d", job_id, len(request.leads), persona.key, persona.version
    )
    return DispatchOut(engine_job_id=job_id, status="accepted")


def decide(
    attempt: Mapping[str, Any], *, connected: bool, outcome: str, now: datetime
) -> tuple[str, str, datetime | None]:
    """What a finished attempt means for its lead: (row status, state reported to the dashboard, next attempt)."""
    if connected and outcome not in RETRY_OUTCOMES:
        return "done", "done", None
    if attempt["job_status"] == "cancelled":
        return "pending", "queued", None
    if attempt["attempts"] >= attempt["max_tries"]:
        return "failed", "exhausted", None
    return "pending", "retry", now + timedelta(minutes=attempt["retry_gap_minutes"])


async def settle_attempt(
    svc: Services,
    attempt: Mapping[str, Any],
    *,
    call_id: int | None,
    connected: bool,
    outcome: str,
    seconds: float,
) -> None:
    now = datetime.now(UTC)
    status, lead_state, next_attempt_at = decide(attempt, connected=connected, outcome=outcome, now=now)
    data: dict[str, Any] = {
        "leadId": attempt["lead_id"],
        "leadPhone": attempt["phone"],
        "callerNumber": svc.settings.exotel_caller_id,
        "outcome": outcome,
        "durationSeconds": round(seconds),
        "attempt": attempt["attempts"],
        "leadState": lead_state,
    }
    if call_id is not None:
        data["engineCallId"] = str(call_id)
        if connected:
            data["transcriptRef"] = str(call_id)  # an opaque key, not a link; see app/reporting.py
    if next_attempt_at is not None:
        data["nextAttemptAt"] = next_attempt_at.isoformat()
    event = {
        "eventId": str(uuid.uuid4()),
        "eventType": "call.completed" if connected else "call.failed",
        "engineJobId": str(attempt["job_id"]),
        "occurredAt": now.isoformat(),
        "data": data,
    }
    await dispatch_db.settle_lead(
        svc.pool,
        lead_pk=attempt["lead_pk"],
        status=status,
        next_attempt_at=next_attempt_at,
        outcome=outcome,
        event=event,
        job_id=attempt["job_id"],
    )


async def abandon_stale_calls(svc: Services) -> None:
    abandoned = await dispatch_db.abandon_stale_calls(
        svc.pool, RINGING_GRACE_SECONDS, svc.settings.max_call_duration_seconds + CONNECTED_GRACE_SECONDS
    )
    released = await dispatch_db.release_unplaced_claims(svc.pool, UNPLACED_CLAIM_SECONDS)
    if abandoned or released:
        log.warning("dispatcher recovered %d stale call(s) and %d unplaced claim(s)", abandoned, released)


async def settle_finished_calls(svc: Services) -> None:
    for row in await dispatch_db.finished_attempts(svc.pool):
        connected = row["call_status"] == "completed" and row["connected_seconds"] is not None
        if connected:
            outcome = row["outcome"] or "completed"
        else:
            outcome = "not_connected" if row["call_status"] == "completed" else row["call_status"]
        await settle_attempt(
            svc,
            row,
            call_id=row["call_id"],
            connected=connected,
            outcome=outcome,
            seconds=row["connected_seconds"] or 0.0,
        )


async def place_next_call(svc: Services) -> bool:
    """Start the next due call if there is capacity. Returns whether a lead was taken."""
    if not svc.settings.enable_outbound_calls or not calls.capacity_left(svc):
        return False
    lead = await dispatch_db.next_due_lead(svc.pool)
    if lead is None or not await dispatch_db.claim_lead(svc.pool, lead["lead_pk"]):
        return False
    attempt = {**lead, "attempts": lead["attempts"] + 1}
    agent = Agent(**dict(await db.get_agent(svc.pool, lead["agent_id"])))
    try:
        call_id = await calls.place_outbound_call(
            svc,
            agent,
            lead["phone"],
            lead["name"],
            lead["lead_pk"],
            voice_id=lead["voice_id"],
            tts_model_id=lead["tts_model_id"],
        )
    except CallRejected:
        await dispatch_db.release_lead(svc.pool, lead["lead_pk"])
        return False
    except ExotelError as exc:
        log.warning("could not dial lead %s: %s", lead["lead_id"], exc)
        unknown = exc.placement_unknown
        await settle_attempt(
            svc,
            {**attempt, "max_tries": attempt["attempts"]} if unknown else attempt,
            call_id=None, connected=False, outcome="placement_unknown" if unknown else "dial_failed", seconds=0.0,
        )
        return True
    await dispatch_db.mark_job_running(svc.pool, lead["job_id"])
    await dispatch_db.enqueue_event(
        svc.pool,
        lead["job_id"],
        {
            "eventId": str(uuid.uuid4()),
            "eventType": "call.started",
            "engineJobId": str(lead["job_id"]),
            "occurredAt": datetime.now(UTC).isoformat(),
            "data": {
                "engineCallId": str(call_id),
                "leadId": lead["lead_id"],
                "leadPhone": lead["phone"],
                "callerNumber": svc.settings.exotel_caller_id,
            },
        },
    )
    return True


async def finalize_jobs(svc: Services) -> None:
    now = datetime.now(UTC).isoformat()
    for job in await dispatch_db.finishable_jobs(svc.pool):
        if await dispatch_db.finalize_job(svc.pool, job["id"], str(uuid.uuid4()), now):
            log.info("dispatch job %s finished", job["id"])


def retry_delay(attempts: int) -> float:
    """Seconds to wait before redelivering an event that has failed `attempts` times."""
    return min(5.0 * 2 ** (attempts - 1), 900.0)


def sign(body: bytes, secret: bytes) -> str:
    return hmac.new(secret, body, hashlib.sha256).hexdigest()


async def _deliver(svc: Services, secret: bytes, row: Mapping[str, Any]) -> bool:
    body = dispatch_db.event_body(row["payload"])
    try:
        response = await svc.http.post(
            row["webhook_url"],
            content=body,
            headers={"content-type": "application/json", "x-voice-engine-signature": sign(body, secret)},
            timeout=WEBHOOK_TIMEOUT_SECONDS,
        )
        status_code: int | None = response.status_code
        error = None if response.is_success else f"HTTP {response.status_code}"
    except httpx.HTTPError as exc:
        status_code, error = None, type(exc).__name__
    if error is None:
        await dispatch_db.mark_delivered(svc.pool, row["id"])
        return True
    if row["job_id"] is None:  # an inbound call report: bounded, and a refusal is final
        refused = status_code is not None and 400 <= status_code < 500 and status_code not in RETRYABLE_CLIENT_ERRORS
        if refused or row["attempts"] + 1 >= svc.settings.webhook_max_attempts:
            log.error("giving up on call report %s (%s)", row["payload"].get("eventId"), error)
            await dispatch_db.mark_abandoned(svc.pool, row["id"], error)
            return True
    log.warning("webhook event %s not delivered (%s); will retry", row["payload"].get("eventId"), error)
    await dispatch_db.mark_delivery_failed(svc.pool, row["id"], error, retry_delay(row["attempts"] + 1))
    return False


async def deliver_webhooks(svc: Services) -> None:
    secret = svc.settings.dashboard_webhook_secret
    if secret is None:
        return
    key = secret.get_secret_value().encode()
    gate = asyncio.Semaphore(MAX_PARALLEL_DELIVERIES)

    async def deliver(row: Mapping[str, Any]) -> bool:
        async with gate:
            return await _deliver(svc, key, row)

    for _ in range(MAX_DELIVERY_ROUNDS):
        heads = await dispatch_db.due_outbox_heads(svc.pool)
        if not heads or not any(await asyncio.gather(*(deliver(row) for row in heads))):
            return


async def tick(svc: Services) -> None:
    steps = (abandon_stale_calls, settle_finished_calls, place_all_due_calls, finalize_jobs)
    for step in steps:
        await _guarded(svc, step)


async def place_all_due_calls(svc: Services) -> None:
    while await place_next_call(svc):
        pass


async def _guarded(svc: Services, step: Callable[[Services], Awaitable[None]]) -> None:
    try:
        await step(svc)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("dispatcher step %s failed", step.__name__)  # type: ignore[attr-defined]


async def _repeat(svc: Services, step: Callable[[Services], Awaitable[None]]) -> None:
    while True:
        await _guarded(svc, step)
        await asyncio.sleep(svc.settings.dispatcher_poll_seconds)


async def run_forever(svc: Services) -> None:
    """Places calls and settles results in one loop, and delivers webhooks in another so that a slow
    dashboard can delay neither dialling nor the saving of a call."""
    await asyncio.gather(_repeat(svc, tick), _repeat(svc, deliver_webhooks))
