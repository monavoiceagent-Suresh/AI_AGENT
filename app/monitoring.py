"""M3 Super Admin operational visibility: a read-only snapshot of the M2 allocator's live state —
overall occupancy, per-tenant breakdown and outbound queue depth, and a couple of "this should
never be non-zero" health counters. Nothing here writes anything; it exists purely to answer
GET /capacity.
"""

import asyncpg

from app.allocator import ChannelAllocator
from app.config import Settings
from app.models import CapacityOut, TenantCapacityOut

# How long a 'reserved'/'releasing' row may sit without moving before it counts as stuck for
# operator visibility — generously longer than any healthy transition, including a slow Exotel
# lookup during reconciliation.
STALE_THRESHOLD_SECONDS = 120


async def capacity_snapshot(pool: asyncpg.Pool, settings: Settings, allocator: ChannelAllocator) -> CapacityOut:
    async with pool.acquire() as conn, conn.transaction(isolation="repeatable_read", readonly=True):
        return await _snapshot(conn, settings)


async def _snapshot(pool, settings) -> CapacityOut:

    by_state = await pool.fetch(
        "SELECT state, count(*) AS n FROM channel_allocations WHERE state != 'released' GROUP BY state"
    )
    counts = {row["state"]: row["n"] for row in by_state}

    by_direction = await pool.fetch(
        "SELECT direction, count(*) AS n FROM channel_allocations WHERE state != 'released' GROUP BY direction"
    )
    direction_counts = {row["direction"]: row["n"] for row in by_direction}

    health = await pool.fetchrow(
        "SELECT "
        "count(*) FILTER (WHERE state = 'reserved' AND reserved_at < now() - make_interval(secs => $1)) "
        "AS stale_reserved, "
        "count(*) FILTER (WHERE state = 'releasing' AND updated_at < now() - make_interval(secs => $1)) "
        "AS stuck_releasing "
        "FROM channel_allocations WHERE state != 'released'",
        STALE_THRESHOLD_SECONDS,
    )

    # New reservations carry their tenant before dialing; legacy rows fall back to calls.
    tenant_rows = await pool.fetch(
        "SELECT COALESCE(ca.client_id, a.client_id) AS client_id, ca.state, count(*) AS n FROM channel_allocations ca "
        "LEFT JOIN calls c ON c.exotel_call_sid = ca.call_sid "
        "LEFT JOIN agents a ON a.id = c.agent_id "
        "WHERE ca.state != 'released' AND COALESCE(ca.client_id, a.client_id) IS NOT NULL "
        "GROUP BY COALESCE(ca.client_id, a.client_id), ca.state"
    )
    queue_rows = await pool.fetch(
        "SELECT j.client_id, count(*) AS n FROM dispatch_leads l JOIN dispatch_jobs j ON j.id = l.job_id "
        "JOIN tenants tenant ON tenant.id = j.client_id AND tenant.status = 'active' "
        "WHERE l.status = 'pending' AND l.next_attempt_at <= now() AND j.status IN ('queued', 'running') "
        "AND l.attempts < j.max_tries "
        "AND to_char(now() AT TIME ZONE 'Asia/Kolkata', 'HH24:MI') BETWEEN j.window_start AND j.window_end "
        "GROUP BY j.client_id"
    )

    tenants: dict[str, dict[str, int]] = {}
    for row in tenant_rows:
        entry = tenants.setdefault(row["client_id"], {"connected": 0, "reserved": 0, "queued_leads": 0})
        state = "reserved" if row["state"] == "releasing" else row["state"]
        entry[state] += row["n"]
    for row in queue_rows:
        entry = tenants.setdefault(row["client_id"], {"connected": 0, "reserved": 0, "queued_leads": 0})
        entry["queued_leads"] = row["n"]

    return CapacityOut(
        max_concurrent_calls=settings.max_concurrent_calls,
        inbound_reserved_calls=settings.inbound_reserved_calls,
        occupied=sum(counts.values()),
        connected=counts.get("connected", 0),
        reserved=counts.get("reserved", 0) + counts.get("releasing", 0),
        inbound_occupied=direction_counts.get("inbound", 0),
        outbound_occupied=direction_counts.get("outbound", 0),
        stale_reserved=health["stale_reserved"],
        stuck_releasing=health["stuck_releasing"],
        unattributed=sum(counts.values()) - sum(t["reserved"] + t["connected"] for t in tenants.values()),
        tenants=[TenantCapacityOut(client_id=client_id, **values) for client_id, values in sorted(tenants.items())],
    )
