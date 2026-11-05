"""Engine-side tenant identity: validates every client_id the engine is asked to act on against a
record synced here by Mona (PUT /tenants), instead of trusting whatever string a request carries.

Three-stage rollout (see Settings.tenant_validation_mode):
  1. "log_only" (the default): an unrecognised or suspended client_id is logged, never blocked.
     Lets an operator see what Mona would actually send before anything can reject it.
  2. Mona backfills every existing Client to this table (see scripts in the Mona repo) and keeps
     syncing on create/suspend/reactivate, so the engine's view is complete before enforcement.
  3. "enforcing": set TENANT_VALIDATION_MODE=enforcing and restart. From then on, every call site
     below raises instead of logging.

Rollback: set TENANT_VALIDATION_MODE back to log_only and restart. Nothing here persists the mode
anywhere but process settings, so a restart is the entire rollback — no migration to undo, no data
to repair, because log_only never rejected anything in the first place.
"""

import logging

import asyncpg

from app.calls import Services

log = logging.getLogger(__name__)


class TenantNotRecognized(Exception):
    """client_id is not a tenant the engine has been told about."""


class TenantSuspended(Exception):
    """client_id belongs to a tenant Mona has marked suspended."""


async def upsert_tenant(pool: asyncpg.Pool, tenant_id: str, name: str, status: str) -> None:
    await pool.execute(
        "INSERT INTO tenants (id, name, status, synced_at) VALUES ($1, $2, $3, now()) "
        "ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, status = EXCLUDED.status, synced_at = now()",
        tenant_id,
        name,
        status,
    )


async def get_tenant(pool: asyncpg.Pool, tenant_id: str) -> asyncpg.Record | None:
    return await pool.fetchrow("SELECT id, name, status, synced_at FROM tenants WHERE id = $1", tenant_id)


async def require_known(svc: Services, client_id: str | None, *, context: str, enforce: bool = False) -> None:
    """Raise if client_id is not a known, active tenant and the engine is enforcing; otherwise log.

    client_id=None is always allowed: it means "no tenant" (the engine-wide default agent), not an
    unrecognised tenant.
    """
    if client_id is None:
        return
    tenant = await get_tenant(svc.pool, client_id)
    enforcing = enforce or svc.settings.tenant_validation_mode == "enforcing"
    if tenant is None:
        message = f"{context}: client_id {client_id!r} is not a tenant the engine has been told about"
        if enforcing:
            raise TenantNotRecognized(message)
        log.warning("tenant validation (log-only): %s", message)
        return
    if tenant["status"] == "suspended":
        message = f"{context}: client_id {client_id!r} is suspended"
        if enforcing:
            raise TenantSuspended(message)
        log.warning("tenant validation (log-only): %s", message)
