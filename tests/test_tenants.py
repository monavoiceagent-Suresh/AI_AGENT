"""Engine-side tenant validation (app/tenants.py): log-only vs enforcing, and the upsert/lookup
queries behind it. Uses a tiny fake pool rather than the docker-compose database: this module's
logic is pure branching over one row, not anything worth a real-database integration test."""

from types import SimpleNamespace

import pytest

from app import tenants
from tests.fakes import make_settings

pytestmark = pytest.mark.anyio


class FakePool:
    def __init__(self, tenant: dict | None = None) -> None:
        self.tenant = tenant
        self.upserts: list[tuple] = []

    async def fetchrow(self, _query: str, tenant_id: str) -> dict | None:
        return self.tenant if self.tenant and self.tenant["id"] == tenant_id else None

    async def execute(self, query: str, *args: object) -> None:
        self.upserts.append((query, args))


def services(pool: FakePool, mode: str = "log_only") -> SimpleNamespace:
    return SimpleNamespace(settings=make_settings(tenant_validation_mode=mode), pool=pool)


async def test_a_call_site_scoped_to_no_tenant_is_always_allowed():
    svc = services(FakePool(), mode="enforcing")
    await tenants.require_known(svc, None, context="test")  # must not raise


async def test_log_only_never_raises_for_an_unknown_or_suspended_tenant(caplog):
    caplog.set_level("WARNING")
    svc = services(FakePool(), mode="log_only")
    await tenants.require_known(svc, "ghost", context="inbound route publish")
    assert any("not a tenant the engine has been told about" in r.getMessage() for r in caplog.records)

    suspended = services(FakePool({"id": "t1", "status": "suspended"}), mode="log_only")
    await tenants.require_known(suspended, "t1", context="campaign dispatch")
    assert any("is suspended" in r.getMessage() for r in caplog.records)


async def test_enforcing_rejects_an_unknown_tenant():
    svc = services(FakePool(), mode="enforcing")
    with pytest.raises(tenants.TenantNotRecognized):
        await tenants.require_known(svc, "ghost", context="test")


async def test_enforcing_rejects_a_suspended_tenant():
    svc = services(FakePool({"id": "t1", "status": "suspended"}), mode="enforcing")
    with pytest.raises(tenants.TenantSuspended):
        await tenants.require_known(svc, "t1", context="test")


async def test_enforcing_allows_a_known_active_tenant():
    svc = services(FakePool({"id": "t1", "status": "active"}), mode="enforcing")
    await tenants.require_known(svc, "t1", context="test")  # must not raise


async def test_upsert_is_idempotent_by_construction():
    """Not a real uniqueness test (that needs Postgres' ON CONFLICT); this only pins the call
    shape so a future edit can't silently drop the upsert semantics from the SQL text."""
    pool = FakePool()
    await tenants.upsert_tenant(pool, "t1", "Acme", "active")
    await tenants.upsert_tenant(pool, "t1", "Acme Renamed", "suspended")
    assert len(pool.upserts) == 2
    assert all("ON CONFLICT" in query for query, _args in pool.upserts)
    assert pool.upserts[1][1] == ("t1", "Acme Renamed", "suspended")


async def test_admission_enforces_status_even_during_log_only_rollout():
    for record, expected in [
        (None, tenants.TenantNotRecognized),
        ({"id": "t1", "status": "suspended"}, tenants.TenantSuspended),
    ]:
        with pytest.raises(expected):
            await tenants.require_known(services(FakePool(record)), "t1", context="admission", enforce=True)
