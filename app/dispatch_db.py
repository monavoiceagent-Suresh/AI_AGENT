"""Queries for dashboard campaign dispatch: jobs, per-lead state and the webhook outbox."""

import json
from datetime import datetime
from typing import Any

import asyncpg

from app.models import DispatchIn, PersonaIn

OPEN_JOB_STATUSES = ("queued", "running")


class OpenJobExists(Exception):
    """The campaign already has a job that has not been finalized."""


PERSONA_FIELDS = ("name", "language", "workflow", "instructions", "greeting", "knowledge")


async def job_by_key(pool: asyncpg.Pool, idempotency_key: str) -> asyncpg.Record | None:
    return await pool.fetchrow("SELECT id, status FROM dispatch_jobs WHERE idempotency_key = $1", idempotency_key)


async def upsert_persona_agent(conn: asyncpg.Connection, client_id: str, persona: PersonaIn) -> int:
    """The agent row for (client, persona key): created if missing, its text refreshed, its voice settings kept.

    Calls never read this text (they use the job's copy); the row exists to hold voice and model overrides
    and to give calls an agent id.
    """
    return await conn.fetchval(
        "INSERT INTO agents (name, language, workflow, instructions, greeting, knowledge, client_id, persona_key) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8) "
        "ON CONFLICT (client_id, persona_key) WHERE persona_key IS NOT NULL DO UPDATE SET "
        "name = EXCLUDED.name, workflow = EXCLUDED.workflow, instructions = EXCLUDED.instructions, "
        "greeting = EXCLUDED.greeting, knowledge = EXCLUDED.knowledge, "
        "voice_id = CASE WHEN agents.language <> EXCLUDED.language THEN NULL ELSE agents.voice_id END, "
        "tts_model_id = CASE WHEN agents.language <> EXCLUDED.language THEN NULL ELSE agents.tts_model_id END, "
        "language = EXCLUDED.language, updated_at = now() RETURNING id",
        persona.name,
        persona.language,
        persona.workflow,
        persona.instructions,
        persona.greeting,
        persona.knowledge,
        client_id,
        persona.key,
    )


async def get_job_persona(pool: asyncpg.Pool, dispatch_lead_id: int) -> dict[str, Any] | None:
    """The persona copy of the job a lead belongs to (None for jobs created without one)."""
    return await pool.fetchval(
        "SELECT j.persona FROM dispatch_leads l JOIN dispatch_jobs j ON j.id = l.job_id WHERE l.id = $1",
        dispatch_lead_id,
    )


async def create_job(pool: asyncpg.Pool, request: DispatchIn) -> str:
    """Insert the job and its leads atomically; raises OpenJobExists if the campaign has an unfinished job."""
    try:
        async with pool.acquire() as conn, conn.transaction():
            agent_id = await upsert_persona_agent(conn, request.client_id, request.persona)
            job_id = await conn.fetchval(
                "INSERT INTO dispatch_jobs (idempotency_key, client_id, campaign_id, agent_id, status, window_start, "
                "window_end, max_tries, retry_gap_minutes, webhook_url, persona) "
                "VALUES ($1, $2, $3, $4, 'queued', $5, $6, $7, $8, $9, $10) RETURNING id",
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
            )
            await conn.executemany(
                "INSERT INTO dispatch_leads (job_id, lead_id, phone, name) VALUES ($1, $2, $3, $4)",
                [(job_id, lead.id, lead.phone, lead.name) for lead in request.leads],
            )
    except asyncpg.UniqueViolationError as exc:
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
    """The next lead to call: due, its job open, and the current India time inside the job's calling window."""
    return await pool.fetchrow(
        "SELECT l.id AS lead_pk, l.lead_id, l.phone, l.name, l.attempts, j.id AS job_id, j.status AS job_status, "
        "j.agent_id, j.max_tries, j.retry_gap_minutes "
        "FROM dispatch_leads l JOIN dispatch_jobs j ON j.id = l.job_id "
        "WHERE l.status = 'pending' AND l.next_attempt_at <= now() AND j.status IN ('queued', 'running') "
        "AND to_char(now() AT TIME ZONE 'Asia/Kolkata', 'HH24:MI') BETWEEN j.window_start AND j.window_end "
        "ORDER BY l.next_attempt_at, l.id LIMIT 1"
    )


async def claim_lead(pool: asyncpg.Pool, lead_pk: int) -> bool:
    claimed = await pool.fetchval(
        "UPDATE dispatch_leads SET status = 'calling', attempts = attempts + 1, claimed_at = now() "
        "WHERE id = $1 AND status = 'pending' RETURNING id",
        lead_pk,
    )
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


async def abandon_stale_calls(pool: asyncpg.Pool, ringing_seconds: int, connected_seconds: int) -> int:
    """Close dispatch calls that never reported back (lost status callback, engine restarted mid-call)."""
    result = await pool.execute(
        "UPDATE calls SET status = 'abandoned', end_reason = 'dispatcher_timeout', ended_at = now() "
        "WHERE dispatch_lead_id IS NOT NULL AND ("
        "(status = 'initiated' AND created_at < now() - make_interval(secs => $1)) OR "
        "(status = 'in_progress' AND started_at < now() - make_interval(secs => $2)))",
        ringing_seconds,
        connected_seconds,
    )
    return int(result.split()[-1])


async def release_unplaced_claims(pool: asyncpg.Pool, older_than_seconds: int) -> int:
    """A claim with no call row (crash between claiming and dialling) goes back to pending."""
    result = await pool.execute(
        "UPDATE dispatch_leads l SET status = 'pending', attempts = attempts - 1, claimed_at = NULL "
        "WHERE l.status = 'calling' AND l.claimed_at < now() - make_interval(secs => $1) "
        "AND NOT EXISTS (SELECT 1 FROM calls c WHERE c.dispatch_lead_id = l.id AND c.created_at >= l.claimed_at)",
        older_than_seconds,
    )
    return int(result.split()[-1])


async def finished_attempts(pool: asyncpg.Pool) -> list[asyncpg.Record]:
    """Leads in 'calling' whose latest call has ended."""
    return await pool.fetch(
        "SELECT l.id AS lead_pk, l.lead_id, l.phone, l.attempts, j.id AS job_id, j.status AS job_status, "
        "j.max_tries, j.retry_gap_minutes, c.id AS call_id, c.status AS call_status, c.outcome, "
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
