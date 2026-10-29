import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import asyncpg

from app import reporting
from app.models import AgentIn

log = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

AGENT_COLUMNS = (
    "name, language, workflow, instructions, greeting, knowledge, collect_fields, "
    "voice_id, tts_model_id, llm_model, is_inbound_default, client_id, persona_key"
)
_AGENT_PLACEHOLDERS = ", ".join(f"${n}" for n in range(1, len(AGENT_COLUMNS.split(",")) + 1))
_AGENT_UPDATE_PLACEHOLDERS = ", ".join(f"${n}" for n in range(2, len(AGENT_COLUMNS.split(",")) + 2))
CALL_COLUMNS = (
    "id, agent_id, direction, exotel_call_sid, from_number, to_number, contact_name, status, end_reason, "
    "created_at, started_at, ended_at, connected_seconds::float8 AS connected_seconds, "
    "usage, cost, latency, outcome, outcome_data, summary"
)


async def _init_connection(conn: asyncpg.Connection) -> None:
    await conn.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")


async def create_pool(dsn: str, min_size: int, max_size: int) -> asyncpg.Pool:
    return await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size, init=_init_connection)


async def migrate(pool: asyncpg.Pool) -> list[str]:
    """Apply pending migrations/*.sql in filename order; returns the versions applied."""
    applied: list[str] = []
    async with pool.acquire() as conn, conn.transaction():
        # Serialise concurrent starters.
        await conn.execute("SELECT pg_advisory_xact_lock(727274)")
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        done = {r["version"] for r in await conn.fetch("SELECT version FROM schema_migrations")}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.stem in done:
                continue
            await conn.execute(path.read_text(encoding="utf-8"))
            await conn.execute("INSERT INTO schema_migrations (version) VALUES ($1)", path.stem)
            applied.append(path.stem)
    return applied


def _agent_args(agent: AgentIn) -> tuple[Any, ...]:
    return (
        agent.name,
        agent.language,
        agent.workflow,
        agent.instructions,
        agent.greeting,
        agent.knowledge,
        agent.collect_fields,
        agent.voice_id,
        agent.tts_model_id,
        agent.llm_model,
        agent.is_inbound_default,
        agent.client_id,
        agent.persona_key,
    )


async def _clear_inbound_default(conn: asyncpg.Connection, agent: AgentIn) -> None:
    if agent.is_inbound_default:
        await conn.execute("UPDATE agents SET is_inbound_default = false WHERE is_inbound_default")


async def create_agent(pool: asyncpg.Pool, agent: AgentIn) -> asyncpg.Record:
    async with pool.acquire() as conn, conn.transaction():
        await _clear_inbound_default(conn, agent)
        return await conn.fetchrow(
            f"INSERT INTO agents ({AGENT_COLUMNS}) VALUES ({_AGENT_PLACEHOLDERS}) RETURNING *",
            *_agent_args(agent),
        )


async def update_agent(pool: asyncpg.Pool, agent_id: int, agent: AgentIn) -> asyncpg.Record | None:
    async with pool.acquire() as conn, conn.transaction():
        await _clear_inbound_default(conn, agent)
        return await conn.fetchrow(
            f"UPDATE agents SET ({AGENT_COLUMNS}, updated_at) = "
            f"({_AGENT_UPDATE_PLACEHOLDERS}, now()) WHERE id = $1 RETURNING *",
            agent_id,
            *_agent_args(agent),
        )


async def list_agents(pool: asyncpg.Pool) -> list[asyncpg.Record]:
    return await pool.fetch("SELECT * FROM agents ORDER BY id")


async def get_agent(pool: asyncpg.Pool, agent_id: int) -> asyncpg.Record | None:
    return await pool.fetchrow("SELECT * FROM agents WHERE id = $1", agent_id)


async def get_inbound_default_agent(pool: asyncpg.Pool) -> asyncpg.Record | None:
    """The engine-wide default agent. An agent that belongs to a workspace is never served this way."""
    return await pool.fetchrow("SELECT * FROM agents WHERE is_inbound_default AND client_id IS NULL")


async def create_outbound_call(
    pool: asyncpg.Pool,
    agent_id: int,
    call_sid: str,
    from_number: str,
    to_number: str,
    contact_name: str | None,
    dispatch_lead_id: int | None = None,
) -> int:
    return await pool.fetchval(
        "INSERT INTO calls (agent_id, direction, exotel_call_sid, from_number, to_number, contact_name, status, "
        "dispatch_lead_id) VALUES ($1, 'outbound', $2, $3, $4, $5, 'initiated', $6) RETURNING id",
        agent_id,
        call_sid,
        from_number,
        to_number,
        contact_name,
        dispatch_lead_id,
    )


async def get_call_by_sid(pool: asyncpg.Pool, call_sid: str) -> asyncpg.Record | None:
    return await pool.fetchrow("SELECT * FROM calls WHERE exotel_call_sid = $1", call_sid)


async def start_call(
    pool: asyncpg.Pool,
    *,
    call_id: int | None,
    agent_id: int,
    call_sid: str,
    stream_sid: str,
    from_number: str,
    to_number: str,
    started_at: datetime,
    dialled_number: str | None = None,
) -> int:
    """Mark a call as connected; creates the row for inbound calls (call_id is None)."""
    if call_id is not None:
        await pool.execute(
            "UPDATE calls SET status = 'in_progress', stream_sid = $2, started_at = $3 WHERE id = $1",
            call_id,
            stream_sid,
            started_at,
        )
        return call_id
    return await pool.fetchval(
        "INSERT INTO calls (agent_id, direction, exotel_call_sid, stream_sid, from_number, to_number, "
        "status, started_at, dialled_number) VALUES ($1, 'inbound', $2, $3, $4, $5, 'in_progress', $6, $7) "
        "RETURNING id",
        agent_id,
        call_sid,
        stream_sid,
        from_number,
        to_number,
        started_at,
        dialled_number,
    )


async def finish_call(
    pool: asyncpg.Pool,
    call_id: int,
    *,
    end_reason: str,
    ended_at: datetime,
    connected_seconds: float,
    usage: dict[str, Any],
    cost: dict[str, Any],
    latency: dict[str, Any],
    turns: list[dict[str, Any]],
    outcome: str,
    outcome_data: dict[str, Any],
    summary: str,
    report_url: str | None = None,
) -> None:
    """Save a finished call. With `report_url`, an inbound call's report is queued in the same transaction."""
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(
            "UPDATE calls SET status = 'completed', end_reason = $2, ended_at = $3, connected_seconds = $4, "
            "usage = $5, cost = $6, latency = $7, outcome = $8, outcome_data = $9, summary = $10 WHERE id = $1",
            call_id,
            end_reason,
            ended_at,
            connected_seconds,
            usage,
            cost,
            latency,
            outcome,
            outcome_data,
            summary,
        )
        await conn.executemany(
            "INSERT INTO call_turns (call_id, idx, user_text, agent_text, interrupted, endpointing_ms, "
            "llm_ttft_ms, tts_ttfb_ms, response_latency_ms) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            [
                (
                    call_id,
                    t["idx"],
                    t["user_text"],
                    t["agent_text"],
                    t["interrupted"],
                    t["endpointing_ms"],
                    t["llm_ttft_ms"],
                    t["tts_ttfb_ms"],
                    t["response_latency_ms"],
                )
                for t in turns
            ],
        )
        if report_url:
            try:
                async with conn.transaction():  # a savepoint: a failed report must never undo the saved call
                    await reporting.enqueue_inbound_call_report(conn, call_id, report_url)
            except Exception:
                log.exception("call %s was saved but its report could not be queued", call_id)


async def apply_status_callback(pool: asyncpg.Pool, call_sid: str, status: str) -> bool:
    """Record a terminal Exotel status for calls whose media stream never connected."""
    result = await pool.execute(
        "UPDATE calls SET status = $2, ended_at = now() WHERE exotel_call_sid = $1 AND status = 'initiated'",
        call_sid,
        status,
    )
    return result != "UPDATE 0"


async def list_calls(pool: asyncpg.Pool, limit: int) -> list[asyncpg.Record]:
    return await pool.fetch(f"SELECT {CALL_COLUMNS} FROM calls ORDER BY id DESC LIMIT $1", limit)


async def get_call(pool: asyncpg.Pool, call_id: int) -> tuple[asyncpg.Record, list[asyncpg.Record]] | None:
    call = await pool.fetchrow(f"SELECT {CALL_COLUMNS} FROM calls WHERE id = $1", call_id)
    if call is None:
        return None
    turns = await pool.fetch("SELECT * FROM call_turns WHERE call_id = $1 ORDER BY idx", call_id)
    return call, turns
