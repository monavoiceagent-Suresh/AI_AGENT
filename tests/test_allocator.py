"""M2 channel allocator: concurrency, duplicate reservations/callbacks, failed dispatch, process
restart reconciliation and stale-reservation sweep, against a real Redis and a real throwaway
PostgreSQL database (the allocator has no in-memory fallback to test instead).
"""

import asyncio
import uuid
from dataclasses import dataclass

import asyncpg
import pytest
from redis.asyncio import from_url

from app import db
from app.allocator import ChannelAllocator
from tests.fakes import make_settings
from tests.test_api import _admin_dsn_and_db, _check_redis, _drop

pytestmark = pytest.mark.anyio


@pytest.fixture
async def allocator():
    try:
        base, name = await _admin_dsn_and_db()
    except (OSError, asyncpg.PostgresError) as exc:
        pytest.skip(f"PostgreSQL not reachable (docker compose up -d db): {exc}")
    redis_url = "redis://127.0.0.1:6379/0"
    try:
        await _check_redis(redis_url)
    except Exception as exc:
        pytest.skip(f"Redis not reachable (docker compose up -d redis): {exc}")

    test_dsn = base.rsplit("/", 1)[0] + f"/{name}"
    pool = await db.create_pool(test_dsn, 1, 5)
    await db.migrate(pool)
    redis = from_url(redis_url)
    key_prefix = f"test-{uuid.uuid4().hex[:8]}"
    settings = make_settings(
        max_concurrent_calls=2,
        inbound_reserved_calls=1,
        allocation_reservation_seconds=1,
        redis_key_prefix=key_prefix,
    )
    try:
        yield ChannelAllocator(redis, pool, settings)
    finally:
        await redis.delete(f"{key_prefix}:allocations")
        await redis.aclose()
        await pool.close()
        await _drop(base, name)


def new_id() -> str:
    return f"out-{uuid.uuid4()}"


# ---- concurrency ------------------------------------------------------------------------------


async def test_concurrent_reservations_never_exceed_the_limit(allocator):
    """20 concurrent outbound reservations against a limit of 1 (2 total minus 1 inbound-reserved)
    must grant exactly one and refuse every other — the race the old per-process counter could not
    survive across more than one process, but a single atomic Redis script can."""
    results = await asyncio.gather(*(allocator.reserve(new_id(), "outbound") for _ in range(20)))
    assert sum(results) == 1
    assert await allocator.occupied() == 1


async def test_inbound_reserved_capacity_is_not_available_to_outbound(allocator):
    assert await allocator.reserve(new_id(), "outbound") is True  # takes the 1 unreserved slot
    assert await allocator.reserve(new_id(), "outbound") is False  # no more unreserved slots
    assert await allocator.reserve("in-1", "inbound", call_sid="in-1") is True  # inbound may use the reserved slot


# ---- duplicate reservations / duplicate callbacks ----------------------------------------------


async def test_reserving_the_same_id_twice_is_idempotent_not_a_double_reservation(allocator):
    allocation_id = new_id()
    assert await allocator.reserve(allocation_id, "outbound") is True
    assert await allocator.reserve(allocation_id, "outbound") is True  # e.g. a retried request
    assert await allocator.occupied() == 1


async def test_duplicate_inbound_stream_start_for_the_same_call_sid_is_deduplicated(allocator):
    """Using call_sid as the allocation id (app/calls.py _resolve_inbound) means a retried/duplicate
    stream-start for the same call is granted again rather than counted twice."""
    assert await allocator.reserve("CAdup", "inbound", call_sid="CAdup") is True
    assert await allocator.reserve("CAdup", "inbound", call_sid="CAdup") is True
    assert await allocator.occupied() == 1


async def test_releasing_the_same_allocation_twice_is_a_harmless_no_op(allocator):
    """A provider status callback delivered twice (Exotel does not guarantee exactly-once) must
    not error, and must not affect any other allocation."""
    allocation_id = new_id()
    other = "in-other"  # the fixture's limit allows only 1 outbound slot; use inbound for the other
    await allocator.reserve(allocation_id, "outbound")
    await allocator.reserve(other, "inbound", call_sid=other)
    await allocator.release(allocation_id)
    await allocator.release(allocation_id)  # duplicate callback
    assert await allocator.occupied() == 1  # only `other` remains


async def test_release_by_call_sid_is_idempotent_against_a_duplicate_webhook(allocator):
    allocation_id = new_id()
    await allocator.reserve(allocation_id, "outbound")
    await allocator.set_call_sid(allocation_id, "CAweb")
    await allocator.release_by_call_sid("CAweb")
    await allocator.release_by_call_sid("CAweb")  # duplicate StatusCallback delivery
    assert await allocator.occupied() == 0


# ---- failed dispatch ----------------------------------------------------------------------------


async def test_a_reservation_that_never_gets_a_call_sid_can_still_be_released_directly(allocator):
    """The shape of a failed dial: reserve, then the provider call throws before a call_sid ever
    exists (app/calls.py place_outbound_call's except-release)."""
    allocation_id = new_id()
    assert await allocator.reserve(allocation_id, "outbound") is True
    await allocator.release(allocation_id)  # what place_outbound_call does on any dial exception
    assert await allocator.occupied() == 0
    row = await allocator._pool.fetchrow("SELECT state FROM channel_allocations WHERE id = $1", allocation_id)
    assert row["state"] == "released"


# ---- stale reservation sweep --------------------------------------------------------------------


async def test_a_reservation_past_its_ttl_retains_potentially_active_capacity(allocator):
    stuck = new_id()
    assert await allocator.reserve(stuck, "outbound") is True
    await asyncio.sleep(1.1)  # past the fixture's 1-second reservation_seconds
    # Time alone must never free a potentially active call.
    assert await allocator.reserve(new_id(), "outbound") is False
    assert await allocator.occupied() == 1


async def test_mark_connected_clears_the_ttl_so_a_long_call_is_never_swept(allocator):
    allocation_id = new_id()
    await allocator.reserve(allocation_id, "outbound")
    await allocator.set_call_sid(allocation_id, "CAlong")
    await allocator.mark_connected(allocation_id)
    await asyncio.sleep(1.1)  # well past reservation_seconds, but this allocation is connected now
    assert await allocator.reserve(new_id(), "outbound") is False  # still occupied: not swept
    assert await allocator.occupied() == 1


# ---- process restart reconciliation --------------------------------------------------------------


@dataclass
class FakeCallRecord:
    status: str


class FakeExotel:
    def __init__(self, records: dict[str, str]) -> None:
        self.records = records  # call_sid -> status

    async def get_call_record(self, call_sid: str, *, wait_seconds: float):
        if call_sid not in self.records:
            raise RuntimeError("not found")
        return FakeCallRecord(status=self.records[call_sid])


async def test_reconcile_restores_a_call_exotel_still_reports_in_progress(allocator):
    allocation_id = new_id()
    await allocator._pool.execute(
        "INSERT INTO channel_allocations (id, direction, call_sid, state) "
        "VALUES ($1, 'outbound', 'CAlive', 'connected')",
        allocation_id,
    )
    # Simulate Redis having been flushed/restarted independently of Postgres.
    await allocator._redis.delete(allocator._hash_key)

    await allocator.reconcile_on_startup(FakeExotel({"CAlive": "in-progress"}))

    assert await allocator.occupied() == 1
    row = await allocator._pool.fetchrow("SELECT state FROM channel_allocations WHERE id = $1", allocation_id)
    assert row["state"] == "connected"


async def test_reconcile_releases_a_call_exotel_reports_already_ended(allocator):
    allocation_id = new_id()
    await allocator._pool.execute(
        "INSERT INTO channel_allocations (id, direction, call_sid, state) "
        "VALUES ($1, 'outbound', 'CAdone', 'connected')",
        allocation_id,
    )

    await allocator.reconcile_on_startup(FakeExotel({"CAdone": "completed"}))

    assert await allocator.occupied() == 0
    row = await allocator._pool.fetchrow("SELECT state FROM channel_allocations WHERE id = $1", allocation_id)
    assert row["state"] == "released"


async def test_reconcile_retains_a_reservation_with_no_call_sid(allocator):
    """A reserved allocation with no call_sid can only mean its own place_outbound_call /
    _resolve_inbound coroutine was killed by this very restart — nothing will ever dial, fail, or
    release it on its own, however recently it was reserved, so reconcile must not keep it."""
    allocation_id = new_id()
    await allocator._pool.execute(
        "INSERT INTO channel_allocations (id, direction, call_sid, state) VALUES ($1, 'outbound', NULL, 'reserved')",
        allocation_id,
    )

    await allocator.reconcile_on_startup(FakeExotel({}))

    assert await allocator.occupied() == 1
    row = await allocator._pool.fetchrow("SELECT state FROM channel_allocations WHERE id = $1", allocation_id)
    assert row["state"] == "reserved"


async def test_redis_loss_cannot_admit_over_the_durable_limit(allocator):
    assert await allocator.reserve(new_id(), "outbound")
    await allocator._redis.delete(allocator._hash_key)
    results = await asyncio.gather(*(allocator.reserve(new_id(), "outbound") for _ in range(20)))
    assert not any(results)
    assert await allocator.occupied() == 1
    assert await allocator._redis.hlen(allocator._hash_key) == 2  # ledger row plus readiness marker


async def test_provider_error_retains_and_restores_capacity(allocator):
    allocation_id = new_id()
    await allocator.reserve(allocation_id, "outbound", call_sid="CAunknown")
    await allocator._redis.delete(allocator._hash_key)
    await allocator.reconcile_on_startup(FakeExotel({}))
    assert await allocator.occupied() == 1
    assert not await allocator.reserve(new_id(), "outbound")


async def test_restore_failure_blocks_admission(allocator, monkeypatch):
    async def fail(**kwargs):
        raise OSError("Redis unavailable")

    monkeypatch.setattr(allocator, "_restore", fail)
    with pytest.raises(OSError):
        await allocator.reserve(new_id(), "outbound")
    assert await allocator.occupied() == 0
