"""Queries for dashboard campaign dispatch: jobs, per-lead state and the webhook outbox."""

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import asyncpg

from app.models import DispatchIn, PersonaIn

OPEN_JOB_STATUSES = ("queued", "running")


class OpenJobExists(Exception):
    """The campaign already has a job that has not been finalized."""


PERSONA_FIELDS = (
    "name",
    "language",
    "workflow",
    "instructions",
    "greeting",
    "knowledge",
    "outcome_rules",
    "outcome_fallback",
    "asr_provider",
    "asr_fallback_provider",
    "tts_provider",
    "tts_fallback_provider",
)
# These default to None on both AgentIn and PersonaIn, so a persona JSON blob saved before they existed
# (an old route or job) simply has none chosen, exactly like a fresh one with no override at all.
PERSONA_OPTIONAL_FIELDS = frozenset({"asr_provider", "asr_fallback_provider", "tts_provider", "tts_fallback_provider"})


async def job_by_key(pool: asyncpg.Pool, idempotency_key: str) -> asyncpg.Record | None:
    return await pool.fetchrow("SELECT id, status FROM dispatch_jobs WHERE idempotency_key = $1", idempotency_key)


async def upsert_persona_agent(conn: asyncpg.Connection, client_id: str, persona: PersonaIn) -> int:
    """The agent row for (client, persona key): created if missing, its text refreshed.

    The row's voice is never set here. A campaign or test call's voice belongs to that call (see calls.py), so
    two campaigns sharing a persona cannot change each other's voice. The voice is cleared only when the
    language changes, because the old voice cannot speak the new language.
    """
    return await conn.fetchval(
        "INSERT INTO agents (name, language, workflow, instructions, greeting, knowledge, outcome_rules, "
        "outcome_fallback, asr_provider, asr_fallback_provider, tts_provider, tts_fallback_provider, "
        "client_id, persona_key) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14) "
        "ON CONFLICT (client_id, persona_key) WHERE persona_key IS NOT NULL DO UPDATE SET "
        "name = EXCLUDED.name, workflow = EXCLUDED.workflow, instructions = EXCLUDED.instructions, "
        "greeting = EXCLUDED.greeting, knowledge = EXCLUDED.knowledge, "
        "outcome_rules = EXCLUDED.outcome_rules, outcome_fallback = EXCLUDED.outcome_fallback, "
        "asr_provider = EXCLUDED.asr_provider, asr_fallback_provider = EXCLUDED.asr_fallback_provider, "
        "tts_provider = EXCLUDED.tts_provider, tts_fallback_provider = EXCLUDED.tts_fallback_provider, "
        "voice_id = CASE WHEN agents.language <> EXCLUDED.language THEN NULL ELSE agents.voice_id END, "
        "tts_model_id = CASE WHEN agents.language <> EXCLUDED.language THEN NULL ELSE agents.tts_model_id END, "
        "language = EXCLUDED.language, updated_at = now() RETURNING id",
        persona.name,
        persona.language,
        persona.workflow,
        persona.instructions,
        persona.greeting,
        persona.knowledge,
        [r.model_dump() for r in persona.outcome_rules],
        persona.outcome_fallback,
        persona.asr_provider,
        persona.asr_fallback_provider,
        persona.tts_provider,
        persona.tts_fallback_provider,
        client_id,
        persona.key,
    )


async def get_job_runtime(pool: asyncpg.Pool, dispatch_lead_id: int) -> asyncpg.Record | None:
    """The job a lead belongs to: its persona copy (None for jobs created without one) and campaign."""
    return await pool.fetchrow(
        "SELECT j.persona, j.campaign_id FROM dispatch_leads l JOIN dispatch_jobs j ON j.id = l.job_id WHERE l.id = $1",
        dispatch_lead_id,
    )


async def create_job(pool: asyncpg.Pool, request: DispatchIn) -> str:
    """Insert the job and its leads atomically; raises OpenJobExists if the campaign has an unfinished job."""
    try:
        async with pool.acquire() as conn, conn.transaction():
            agent_id = await upsert_persona_agent(conn, request.client_id, request.persona)
            job_id = await conn.fetchval(
                "INSERT INTO dispatch_jobs (idempotency_key, client_id, campaign_id, agent_id, status, window_start, "
                "window_end, max_tries, retry_gap_minutes, webhook_url, persona, voice_id, tts_model_id, caller_id) "
                "VALUES ($1, $2, $3, $4, 'queued', $5, $6, $7, $8, $9, $10, $11, $12, $13) RETURNING id",
                request.idempotency_key,
                request.client_id,
                request.campaign_id,
                agent_id,
                request.calling_window.start,
                request.calling_window.end,
                request.retry_policy.max_tries,
                request.retry_policy.retry_gap_minutes,
                request.webhook_url,
                request.persona.model_dump(),
                request.voice_id,
                request.tts_model_id,
                request.caller_id,
            )
            await conn.executemany(
                "INSERT INTO dispatch_leads (job_id, lead_id, phone, name, next_attempt_at, attempts) "
                "VALUES ($1, $2, $3, $4, COALESCE($5, now()), $6)",
                [
                    (
                        job_id,
                        lead.id,
                        lead.phone,
                        lead.name,
                        max((d for d in (request.not_before, lead.next_attempt_at) if d is not None), default=None),
                        lead.attempts,
                    )
                    for lead in request.leads
                ],
            )
    except asyncpg.UniqueViolationError as exc:
        # Concurrent retries can both miss job_by_key before either transaction commits.
        existing = await job_by_key(pool, request.idempotency_key)
        if existing is not None:
            return str(existing["id"])
        if exc.constraint_name == "dispatch_jobs_one_open_per_campaign":
            raise OpenJobExists from exc
        raise
    return str(job_id)


async def get_job(pool: asyncpg.Pool, job_id: str) -> asyncpg.Record | None:
    return await pool.fetchrow(
        "SELECT j.id, j.status, "
        "count(*) FILTER (WHERE l.status = 'done')::int AS calls_completed, "
        "count(*) FILTER (WHERE l.status = 'failed')::int AS calls_failed "
        "FROM dispatch_jobs j LEFT JOIN dispatch_leads l ON l.job_id = j.id WHERE j.id = $1 GROUP BY j.id",
        job_id,
    )


async def cancel_job(pool: asyncpg.Pool, job_id: str) -> str | None:
    """Stop placing calls for a job; returns its status afterwards, or None if it does not exist."""
    await pool.execute(
        "UPDATE dispatch_jobs SET status = 'cancelled' WHERE id = $1 AND status IN ('queued', 'running')", job_id
    )
    return await pool.fetchval("SELECT status FROM dispatch_jobs WHERE id = $1", job_id)


# ---- placing calls ---------------------------------------------------------------------------


async def next_due_lead(pool: asyncpg.Pool) -> asyncpg.Record | None:
    """Choose the least-loaded tenant, then the least recently served tenant.

    Persistent service history prevents oldest-backlog starvation when loads return to zero.
    The dispatcher remains single-owner; claim_lead uses compare-and-set for duplicate claims.
    """
    return await pool.fetchrow(
        "WITH tenant_load AS ("
        "    SELECT j2.client_id, count(*) AS in_flight FROM dispatch_leads l2 "
        "    JOIN dispatch_jobs j2 ON j2.id = l2.job_id WHERE l2.status = 'calling' GROUP BY j2.client_id"
        ") "
        "SELECT l.id AS lead_pk, l.lead_id, l.phone, l.name, l.attempts, j.id AS job_id, j.status AS job_status, "
        "j.agent_id, j.max_tries, j.retry_gap_minutes, j.voice_id, j.tts_model_id, j.caller_id "
        "FROM dispatch_leads l JOIN dispatch_jobs j ON j.id = l.job_id "
        "LEFT JOIN tenant_load t ON t.client_id = j.client_id "
        "JOIN tenants tenant ON tenant.id = j.client_id AND tenant.status = 'active' "
        "WHERE l.status = 'pending' AND l.next_attempt_at <= now() AND j.status IN ('queued', 'running') "
        "AND l.attempts < j.max_tries "
        "AND to_char(now() AT TIME ZONE 'Asia/Kolkata', 'HH24:MI') BETWEEN j.window_start AND j.window_end "
        "ORDER BY COALESCE(t.in_flight, 0) ASC, "
        "(SELECT max(j3.last_served_at) FROM dispatch_jobs j3 WHERE j3.client_id = j.client_id) ASC NULLS FIRST, "
        "l.next_attempt_at ASC, l.id ASC LIMIT 1"
    )


async def claim_lead(pool: asyncpg.Pool, lead_pk: int) -> bool:
    async with pool.acquire() as conn, conn.transaction():
        claimed = await conn.fetchval(
            "UPDATE dispatch_leads SET status = 'calling', attempts = attempts + 1, claimed_at = now() "
            "WHERE id = $1 AND status = 'pending' RETURNING job_id",
            lead_pk,
        )
        if claimed is not None:
            await conn.execute("UPDATE dispatch_jobs SET last_served_at = clock_timestamp() WHERE id = $1", claimed)
    return claimed is not None


async def release_lead(pool: asyncpg.Pool, lead_pk: int) -> None:
    """Undo a claim when the call could not be placed for lack of capacity."""
    await pool.execute(
        "UPDATE dispatch_leads SET status = 'pending', attempts = attempts - 1, claimed_at = NULL "
        "WHERE id = $1 AND status = 'calling'",
        lead_pk,
    )


async def mark_job_running(pool: asyncpg.Pool, job_id: Any) -> None:
    await pool.execute("UPDATE dispatch_jobs SET status = 'running' WHERE id = $1 AND status = 'queued'", job_id)


# ---- reconciling finished calls --------------------------------------------------------------


async def candidate_stale_calls(
    pool: asyncpg.Pool, ringing_seconds: int, connected_seconds: int
) -> list[asyncpg.Record]:
    """Dispatch calls that have gone quiet for longer than a status callback should ever take
    (lost callback, engine restarted mid-call). M3: a SELECT only — the caller (dispatch.py)
    confirms each one with Exotel's own record before declaring it abandoned, so a call that is
    genuinely still connected is never released just because its callback was merely slow
    ("release allocations only on confirmed terminal states").
    """
    return await pool.fetch(
        "SELECT id, exotel_call_sid FROM calls WHERE dispatch_lead_id IS NOT NULL AND ("
        "(status = 'initiated' AND created_at < now() - make_interval(secs => $1)) OR "
        "(status = 'in_progress' AND started_at < now() - make_interval(secs => $2)))",
        ringing_seconds,
        connected_seconds,
    )


async def finalize_abandoned_call(pool: asyncpg.Pool, call_id: int) -> None:
    """Closes one call confirmed (or unconfirmable — see dispatch.py) abandoned. Guarded by
    status: a call that reported back normally between the candidate scan and this call is left
    alone, never overwritten."""
    await pool.execute(
        "UPDATE calls SET status = 'abandoned', end_reason = 'dispatcher_timeout', ended_at = now() "
        "WHERE id = $1 AND status IN ('initiated', 'in_progress')",
        call_id,
    )


async def release_unplaced_claims(pool: asyncpg.Pool, older_than_seconds: int) -> int:
    """Fail closed after a crash with unknown placement; requeueing could dial somebody twice."""
    async with pool.acquire() as conn, conn.transaction():
        rows = await conn.fetch(
            "UPDATE dispatch_leads l SET status = 'failed', last_outcome = 'placement_unknown', claimed_at = NULL "
            "WHERE l.status = 'calling' AND l.claimed_at < now() - make_interval(secs => $1) "
            "AND NOT EXISTS (SELECT 1 FROM calls c WHERE c.dispatch_lead_id = l.id AND c.created_at >= l.claimed_at) "
            "RETURNING l.job_id, l.lead_id, l.phone, l.attempts",
            older_than_seconds,
        )
        for row in rows:
            await enqueue_event(
                conn,
                row["job_id"],
                {
                    "eventId": str(uuid.uuid4()),
                    "eventType": "call.failed",
                    "engineJobId": str(row["job_id"]),
                    "occurredAt": datetime.now(UTC).isoformat(),
                    "data": {
                        "leadId": row["lead_id"],
                        "leadPhone": row["phone"],
                        "attempt": row["attempts"],
                        "outcome": "placement_unknown",
                        "durationSeconds": 0,
                        "leadState": "exhausted",
                    },
                },
            )
    return len(rows)


async def finished_attempts(pool: asyncpg.Pool) -> list[asyncpg.Record]:
    """Leads in 'calling' whose latest call has ended."""
    return await pool.fetch(
        "SELECT l.id AS lead_pk, l.lead_id, l.phone, l.attempts, j.id AS job_id, j.status AS job_status, "
        "j.max_tries, j.retry_gap_minutes, j.caller_id, c.id AS call_id, c.status AS call_status, c.outcome, "
        "c.connected_seconds::float8 AS connected_seconds "
        "FROM dispatch_leads l JOIN dispatch_jobs j ON j.id = l.job_id "
        "JOIN calls c ON c.dispatch_lead_id = l.id "
        "WHERE l.status = 'calling' AND c.status NOT IN ('initiated', 'in_progress') "
        "AND c.id = (SELECT max(id) FROM calls WHERE dispatch_lead_id = l.id)"
    )


async def settle_lead(
    pool: asyncpg.Pool,
    *,
    lead_pk: int,
    status: str,
    next_attempt_at: datetime | None,
    outcome: str,
    event: dict[str, Any],
    job_id: Any,
) -> bool:
    """Record one attempt's result and queue its webhook event. False if another pass already settled it."""
    async with pool.acquire() as conn, conn.transaction():
        settled = await conn.fetchval(
            "UPDATE dispatch_leads SET status = $2, next_attempt_at = COALESCE($3, next_attempt_at), "
            "last_outcome = $4, claimed_at = NULL WHERE id = $1 AND status = 'calling' RETURNING id",
            lead_pk,
            status,
            next_attempt_at,
            outcome,
        )
        if settled is None:
            return False
        await enqueue_event(conn, job_id, event)
        return True


# ---- finishing jobs --------------------------------------------------------------------------


async def finishable_jobs(pool: asyncpg.Pool) -> list[asyncpg.Record]:
    """Open jobs with no call in flight and (unless cancelled) no lead still waiting to be called."""
    return await pool.fetch(
        "SELECT j.id, j.status FROM dispatch_jobs j WHERE j.finalized_at IS NULL "
        "AND NOT EXISTS (SELECT 1 FROM dispatch_leads l WHERE l.job_id = j.id AND l.status = 'calling') "
        "AND (j.status = 'cancelled' OR NOT EXISTS "
        "(SELECT 1 FROM dispatch_leads l WHERE l.job_id = j.id AND l.status = 'pending'))"
    )


async def finalize_job(pool: asyncpg.Pool, job_id: Any, event_id: str, occurred_at: str) -> bool:
    async with pool.acquire() as conn, conn.transaction():
        status = await conn.fetchval(
            "UPDATE dispatch_jobs SET status = CASE WHEN status = 'cancelled' THEN 'cancelled' ELSE 'completed' END, "
            "finalized_at = now() WHERE id = $1 AND finalized_at IS NULL RETURNING status",
            job_id,
        )
        if status is None:
            return False
        seconds = await conn.fetchval(
            "SELECT COALESCE(sum(c.connected_seconds), 0)::float8 FROM calls c "
            "JOIN dispatch_leads l ON l.id = c.dispatch_lead_id WHERE l.job_id = $1",
            job_id,
        )
        counts = await conn.fetchrow(
            "SELECT count(*) FILTER (WHERE status = 'done')::int AS done, "
            "count(*) FILTER (WHERE status = 'failed')::int AS failed FROM dispatch_leads WHERE job_id = $1",
            job_id,
        )
        await enqueue_event(
            conn,
            job_id,
            {
                "eventId": event_id,
                "eventType": "job.cancelled" if status == "cancelled" else "job.completed",
                "engineJobId": str(job_id),
                "occurredAt": occurred_at,
                "data": {
                    "minutesUsed": round(seconds / 60, 2),
                    "leadsDone": counts["done"],
                    "leadsFailed": counts["failed"],
                },
            },
        )
        return True


# ---- webhook outbox --------------------------------------------------------------------------


async def enqueue_event(conn: asyncpg.Connection | asyncpg.Pool, job_id: Any, event: dict[str, Any]) -> None:
    await conn.execute(
        "INSERT INTO webhook_outbox (event_id, job_id, payload) VALUES ($1, $2, $3)", event["eventId"], job_id, event
    )


async def due_outbox_heads(pool: asyncpg.Pool) -> list[asyncpg.Record]:
    """The next event to deliver for each job, if it is due: events of one job are delivered in order.

    Events with no job (inbound call reports) are independent of each other, so each is its own head.
    """
    return await pool.fetch(
        "SELECT * FROM ("
        "SELECT DISTINCT ON (COALESCE(o.job_id::text, 'call-' || o.id::text)) "
        "o.id, o.job_id, o.attempts, o.payload, o.next_attempt_at, "
        "COALESCE(o.webhook_url, j.webhook_url) AS webhook_url "
        "FROM webhook_outbox o LEFT JOIN dispatch_jobs j ON j.id = o.job_id "
        "WHERE o.delivered_at IS NULL AND o.abandoned_at IS NULL "
        "ORDER BY COALESCE(o.job_id::text, 'call-' || o.id::text), o.id) heads "
        "WHERE heads.next_attempt_at <= now() ORDER BY heads.id"
    )


async def mark_delivered(pool: asyncpg.Pool, outbox_id: int) -> None:
    await pool.execute("UPDATE webhook_outbox SET delivered_at = now() WHERE id = $1", outbox_id)


async def mark_abandoned(pool: asyncpg.Pool, outbox_id: int, error: str) -> None:
    """Stop retrying an inbound call report. The row stays, with its last error, for inspection."""
    await pool.execute(
        "UPDATE webhook_outbox SET attempts = attempts + 1, last_error = $2, abandoned_at = now() WHERE id = $1",
        outbox_id,
        error,
    )


async def mark_delivery_failed(pool: asyncpg.Pool, outbox_id: int, error: str, delay_seconds: float) -> None:
    await pool.execute(
        "UPDATE webhook_outbox SET attempts = attempts + 1, last_error = $2, "
        "next_attempt_at = now() + make_interval(secs => $3) WHERE id = $1",
        outbox_id,
        error,
        delay_seconds,
    )


def event_body(payload: dict[str, Any]) -> bytes:
    """The exact bytes that are signed and sent."""
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
