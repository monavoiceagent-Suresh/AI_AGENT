"""Shared allocator: PostgreSQL serializes admission and owns the durable ledger.
Redis is atomically rebuilt before admission; uncertainty always retains capacity.
"""

import json
import logging
import uuid
from typing import Literal

import asyncpg
from redis.asyncio import Redis

from app.config import Settings
from app.providers.exotel import ExotelClient

log = logging.getLogger(__name__)

Direction = Literal["inbound", "outbound"]
State = Literal["reserved", "connected", "releasing", "released"]

# Atomic mirror replacement and admission, always under the PostgreSQL allocator lock.
_RESTORE = """
redis.call('DEL', KEYS[1])
redis.call('HSET', KEYS[1], '__ready__', '1')
for i = 1, #ARGV, 2 do redis.call('HSET', KEYS[1], ARGV[i], ARGV[i+1]) end
return 1
"""
_RESERVE = """
if redis.call('HEXISTS', KEYS[1], '__ready__') == 0 then return -1 end
if redis.call('HEXISTS', KEYS[1], ARGV[1]) == 1 then return 1 end
if redis.call('HLEN', KEYS[1]) - 1 >= tonumber(ARGV[3]) then return 0 end
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
return 1
"""


class ChannelAllocator:
    """`settings` is read live on every call, not captured once at construction — Settings is a
    mutable object elsewhere in this codebase (tests, and in principle a future admin endpoint),
    and MAX_CONCURRENT_CALLS/INBOUND_RESERVED_CALLS changing at runtime must take effect
    immediately, exactly as it did for the old capacity_left.
    """

    def __init__(self, redis: Redis, pool: asyncpg.Pool, settings: Settings) -> None:
        self._redis = redis
        self._pool = pool
        self._settings = settings
        self._hash_key = f"{settings.redis_key_prefix}:allocations"
        self._restore = redis.register_script(_RESTORE)
        self._reserve = redis.register_script(_RESERVE)

    async def _rebuild(self, conn) -> None:
        rows = await conn.fetch("SELECT id, direction FROM channel_allocations WHERE state != 'released'")
        args = [v for row in rows for v in (row["id"], json.dumps({"direction": row["direction"]}))]
        await self._restore(keys=[self._hash_key], args=args)

    async def reserve(
        self,
        allocation_id: str,
        direction: Direction,
        *,
        call_sid: str | None = None,
        client_id: str | None = None,
        route_phone: str | None = None,
        dispatch_lead_id: int | None = None,
    ) -> bool:
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(71632491)")
            if client_id is not None:
                status = await conn.fetchval("SELECT status FROM tenants WHERE id = $1 FOR SHARE", client_id)
                if status != "active":
                    return False
            if route_phone is not None:
                await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", route_phone)
                permitted = await conn.fetchval(
                    "SELECT active FROM inbound_routes WHERE phone = $1 AND client_id = $2",
                    route_phone,
                    client_id,
                )
                if not permitted:
                    return False
            if dispatch_lead_id is not None:
                status = await conn.fetchval(
                    "SELECT j.status FROM dispatch_jobs j JOIN dispatch_leads l ON l.job_id = j.id "
                    "WHERE l.id = $1 FOR SHARE OF j",
                    dispatch_lead_id,
                )
                if status not in {"queued", "running"}:
                    return False
            await self._rebuild(conn)
            state = await conn.fetchval("SELECT state FROM channel_allocations WHERE id = $1", allocation_id)
            if state == "released":
                return False
            limit = self._settings.max_concurrent_calls
            if direction == "outbound":
                limit -= self._settings.inbound_reserved_calls
            granted = await self._reserve(
                keys=[self._hash_key],
                args=[
                    allocation_id,
                    json.dumps({"direction": direction}),
                    limit,
                ],
            )
            if granted != 1:
                return False
            await conn.execute(
                "INSERT INTO channel_allocations (id, direction, call_sid, client_id) VALUES ($1, $2, $3, $4) "
                "ON CONFLICT (id) DO NOTHING",
                allocation_id,
                direction,
                call_sid,
                client_id,
            )
        return True

    async def set_call_sid(self, allocation_id: str, call_sid: str) -> None:
        """Attaches the Exotel CallSid once known (outbound only: inbound already used it as the
        allocation id), so later events that only carry a call_sid can find this allocation."""
        await self._pool.execute(
            "UPDATE channel_allocations SET call_sid = $2, updated_at = now() WHERE id = $1",
            allocation_id,
            call_sid,
        )

    async def mark_connected(self, allocation_id: str) -> None:
        """Reserved -> Connected: the call answered. Holds its slot with no further time limit."""
        await self._pool.execute(
            "UPDATE channel_allocations SET state = 'connected', connected_at = now(), updated_at = now() "
            "WHERE id = $1 AND state = 'reserved'",
            allocation_id,
        )

    async def mark_connected_by_call_sid(self, call_sid: str) -> None:
        allocation_id = await self._id_for_call_sid(call_sid)
        if allocation_id is not None:
            await self.mark_connected(allocation_id)

    async def release(self, allocation_id: str) -> None:
        """Only confirmed terminal calls or confirmed placement rejection may release."""
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(71632491)")
            await conn.execute(
                "UPDATE channel_allocations SET state = 'released', released_at = now(), updated_at = now() "
                "WHERE id = $1 AND state != 'released'",
                allocation_id,
            )
            await self._redis.hdel(self._hash_key, allocation_id)

    async def release_by_call_sid(self, call_sid: str) -> None:
        allocation_id = await self._id_for_call_sid(call_sid)
        if allocation_id is not None:
            await self.release(allocation_id)

    async def _id_for_call_sid(self, call_sid: str) -> str | None:
        return await self._pool.fetchval(
            "SELECT id FROM channel_allocations WHERE call_sid = $1 AND state != 'released'", call_sid
        )

    async def occupied(self) -> int:
        """Current total reserved+connected count, for /health and operational visibility."""
        return await self._pool.fetchval("SELECT count(*) FROM channel_allocations WHERE state != 'released'")

    async def reconcile_on_startup(self, exotel: ExotelClient) -> None:
        """Conservatively reconcile all calls; no SID and provider errors retain capacity."""
        rows = await self._pool.fetch(
            "SELECT id, direction, call_sid FROM channel_allocations WHERE state != 'released'"
        )
        for row in rows:
            if not row["call_sid"]:
                continue
            try:
                record = await exotel.get_call_record(row["call_sid"], wait_seconds=5.0)
            except Exception:
                log.warning("cannot confirm call %s; retaining allocation", row["call_sid"])
                continue
            if record.status.lower() in {"completed", "failed", "busy", "no-answer", "canceled", "cancelled"}:
                await self.release(row["id"])
        async with self._pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(71632491)")
            await self._rebuild(conn)


def new_allocation_id() -> str:
    return f"out-{uuid.uuid4()}"
