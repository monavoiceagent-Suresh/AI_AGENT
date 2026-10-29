"""Queries for inbound routing: the published number-to-persona table."""

from typing import Literal

import asyncpg

from app import dispatch_db
from app.models import InboundRoutePublishIn


class StaleVersion(Exception):
    def __init__(self, current: int) -> None:
        super().__init__(f"the engine already has version {current}")
        self.current = current


class VersionConflict(Exception):
    """The same version was published again with different content."""


class NumberTaken(Exception):
    """The number is routed for a different workspace."""


async def find_route(pool: asyncpg.Pool, phone: str) -> asyncpg.Record | None:
    return await pool.fetchrow(
        "SELECT phone, client_id, agent_id, version, active, persona FROM inbound_routes WHERE phone = $1", phone
    )


async def find_route_for_call(pool: asyncpg.Pool, phone: str) -> asyncpg.Record | None:
    """The route that answers a call to `phone`: the exact number, else the same number saved with a stray trunk 0.

    Routes published before the dashboard rejected `+91 0xxxxxxxxxx` can still be keyed `910xxxxxxxxxx`. No Indian
    number starts with 0 after the country code, so the two forms can never be different numbers.
    """
    keys = [phone]
    if phone.startswith("91") and len(phone) == 12:
        keys.append("910" + phone[2:])
    return await pool.fetchrow(
        "SELECT phone, client_id, agent_id, version, active, persona FROM inbound_routes "
        "WHERE phone = ANY($1::text[]) ORDER BY (phone = $2) DESC LIMIT 1",
        keys,
        phone,
    )


async def note_unrouted(pool: asyncpg.Pool, phone: str | None, raw: str, answered_by: str) -> None:
    """Remember that a call arrived for a number with no published route."""
    await pool.execute(
        "INSERT INTO inbound_unrouted (phone, raw_example, calls, last_answered_by) VALUES ($1, $2, 1, $3) "
        "ON CONFLICT (phone) DO UPDATE SET calls = inbound_unrouted.calls + 1, last_seen = now(), "
        "raw_example = EXCLUDED.raw_example, last_answered_by = EXCLUDED.last_answered_by",
        phone or "unreadable",
        raw[:40],
        answered_by,
    )


async def list_unrouted(pool: asyncpg.Pool, limit: int = 20) -> list[asyncpg.Record]:
    """Numbers calls reached with no route, newest first. A number that has since been routed drops off."""
    return await pool.fetch(
        "SELECT u.phone, u.raw_example, u.calls, u.first_seen, u.last_seen, u.last_answered_by AS answered_by "
        "FROM inbound_unrouted u WHERE NOT EXISTS (SELECT 1 FROM inbound_routes r WHERE r.phone = u.phone) "
        "ORDER BY u.last_seen DESC LIMIT $1",
        limit,
    )


async def publish_route(
    pool: asyncpg.Pool, phone: str, request: InboundRoutePublishIn
) -> Literal["applied", "unchanged"]:
    """Store a published route. Versions only move forward, and a number never changes workspace."""
    persona = request.persona.model_dump()
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", phone)  # serialises publishes per number
        current = await conn.fetchrow(
            "SELECT client_id, version, active, persona FROM inbound_routes WHERE phone = $1", phone
        )
        if current is not None:
            if current["client_id"] != request.client_id:
                raise NumberTaken
            if request.version < current["version"]:
                raise StaleVersion(current["version"])
            if request.version == current["version"]:
                if current["active"] == request.active and current["persona"] == persona:
                    return "unchanged"
                raise VersionConflict
        agent_id = await dispatch_db.upsert_persona_agent(conn, request.client_id, request.persona)
        await conn.execute(
            "INSERT INTO inbound_routes (phone, client_id, agent_id, version, active, persona) "
            "VALUES ($1, $2, $3, $4, $5, $6) "
            "ON CONFLICT (phone) DO UPDATE SET agent_id = EXCLUDED.agent_id, version = EXCLUDED.version, "
            "active = EXCLUDED.active, persona = EXCLUDED.persona, published_at = now()",
            phone,
            request.client_id,
            agent_id,
            request.version,
            request.active,
            persona,
        )
    return "applied"
