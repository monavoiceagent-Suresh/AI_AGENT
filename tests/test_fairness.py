"""M3: fair tenant-aware outbound scheduling (app/dispatch_db.next_due_lead) and inbound admission
priority over new outbound dispatch. Fairness is a SQL ORDER BY, so the core assertions run
directly against it with fixture rows inserted over the real docker-compose database — precise and
deterministic, rather than racing the live async dispatcher loop for the same signal.

The `client`/`ready` fixtures build the app (and its asyncpg pool) on the TestClient's own portal
thread/event loop; queries against that pool must go through `client.portal.call(...)`, exactly
like test_dispatch.py's `run_now` helper — calling the pool directly from this test's own event
loop fails with "attached to a different loop".
"""

from app import dispatch_db
from tests.test_api import ADMIN, EXOTEL_AUTH, client, converse  # noqa: F401
from tests.test_dispatch import dispatch_body, ready, sink, start, wait_until  # noqa: F401
from tests.test_inbound import NUMBER, publish  # noqa: F401


def insert_job_with_leads(client, client_id: str, campaign_id: str, lead_phones: list[str], *, calling: int = 0):  # noqa: F811
    """A minimal dispatch_jobs/dispatch_leads fixture: `calling` leads are pre-claimed (in flight),
    the rest are due now. Returns the inserted dispatch_leads ids in `lead_phones` order."""
    pool = client.app.state.services.pool

    def run():
        async def go():
            await pool.execute(
                "INSERT INTO tenants (id, name, status) VALUES ($1, $1, 'active') ON CONFLICT (id) DO NOTHING",
                client_id,
            )
            agent_id = await pool.fetchval(
                "INSERT INTO agents (name, language, workflow, instructions, greeting, client_id, persona_key) "
                "VALUES ('Fairness test', 'en', 'general', 'x', 'hi', $1, $2) RETURNING id",
                client_id,
                f"persona-{campaign_id}",
            )
            job_id = await pool.fetchval(
                "INSERT INTO dispatch_jobs (idempotency_key, client_id, campaign_id, agent_id, status, "
                "window_start, window_end, max_tries, retry_gap_minutes, webhook_url, persona) "
                "VALUES ($1, $2, $3, $4, 'running', '00:00', '23:59', 5, 30, 'https://x.test/hook', "
                "'{}'::jsonb) RETURNING id",
                f"key-{campaign_id}",
                client_id,
                campaign_id,
                agent_id,
            )
            lead_ids = []
            for i, phone in enumerate(lead_phones):
                status = "calling" if i < calling else "pending"
                lead_id = await pool.fetchval(
                    "INSERT INTO dispatch_leads (job_id, lead_id, phone, name, next_attempt_at, attempts, "
                    "status, claimed_at) VALUES ($1, $2, $3, 'x', now(), 0, $4, "
                    "CASE WHEN $4 = 'calling' THEN now() END) RETURNING id",
                    job_id,
                    f"lead-{campaign_id}-{i}",
                    phone,
                    status,
                )
                lead_ids.append(lead_id)
            return lead_ids

        return go()

    return client.portal.call(run)


def test_a_tenant_with_no_calls_in_flight_is_preferred_over_a_busier_tenant(client):  # noqa: F811
    busy_leads = insert_job_with_leads(client, "tenant-a", "campaign-a", ["+919876500001", "+919876500002"], calling=1)
    idle_leads = insert_job_with_leads(client, "tenant-b", "campaign-b", ["+919876500003"])

    picked = client.portal.call(dispatch_db.next_due_lead, client.app.state.services.pool, True)

    assert picked["lead_pk"] == idle_leads[0]  # tenant B (0 in flight) beats tenant A's due second lead
    assert picked["lead_pk"] != busy_leads[1]


def test_ties_within_a_tenant_still_break_by_how_overdue_the_lead_is(client):  # noqa: F811
    """Fairness only orders ACROSS tenants; within one tenant's own due leads, the existing
    oldest-due-first rule (preserved from before M3) still applies."""
    lead_ids = insert_job_with_leads(client, "tenant-a", "campaign-a", ["+919876500006", "+919876500007"])

    picked = client.portal.call(dispatch_db.next_due_lead, client.app.state.services.pool, True)

    assert picked["lead_pk"] == lead_ids[0]  # inserted (and thus due) first


def test_inbound_is_still_admitted_while_outbound_holds_every_unreserved_slot(ready):  # noqa: F811
    """M3 requirement 1: inbound admission priority over new outbound dispatch, unchanged from
    M2 and re-verified at the dispatcher level. A connected outbound call is never interrupted,
    and a new inbound call is still admitted using its reserved slot."""
    publish(ready)
    ready.app.state.services.settings.max_concurrent_calls = 2
    ready.app.state.services.settings.inbound_reserved_calls = 1
    start(ready, clientId="tenant-a", campaignId="campaign-a", leads=[{"id": "a1", "phone": "+919876500009"}])

    wait_until(lambda: ready.app.state.services.exotel.dialled)
    converse(ready, "CAin-fair", to_number=NUMBER)  # must be admitted, not rejected

    assert ready.portal.call(ready.app.state.services.allocator.occupied) == 2  # media close is not terminal proof
    assert ready.app.state.services.exotel.dialled == ["+919876500009"]  # the outbound call was never touched


def test_single_slot_rotates_across_sustained_backlogs(client):  # noqa: F811
    a = insert_job_with_leads(client, "tenant-a", "rotation-a", ["+919876501001", "+919876501002"])
    b = insert_job_with_leads(client, "tenant-b", "rotation-b", ["+919876501003", "+919876501004"])
    pool = client.app.state.services.pool
    picked = []
    for _ in range(8):
        lead = client.portal.call(dispatch_db.next_due_lead, pool, True)
        picked.append(lead["lead_pk"])
        assert client.portal.call(dispatch_db.claim_lead, pool, lead["lead_pk"])
        client.portal.call(dispatch_db.release_lead, pool, lead["lead_pk"])
    assert all((picked[i] in a) != (picked[i + 1] in a) for i in range(7))
    assert set(picked).issubset(set(a + b))


def test_suspended_tenant_is_excluded_from_queued_dispatch(client):  # noqa: F811
    insert_job_with_leads(client, "tenant-a", "suspended-a", ["+919876501010"])
    pool = client.app.state.services.pool
    client.portal.call(pool.execute, "UPDATE tenants SET status = 'suspended' WHERE id = 'tenant-a'")
    assert client.portal.call(dispatch_db.next_due_lead, pool, True) is None
    picked = client.portal.call(dispatch_db.next_due_lead, pool, False)
    assert picked is not None  # log_only mode never blocks placement on tenant status


def test_suspended_tenant_is_rejected_at_inbound_admission(client):  # noqa: F811
    from tests.test_inbound import refused

    publish(client)
    client.put("/tenants", json={"id": "client-1", "name": "Test", "status": "suspended"}, headers=ADMIN)
    refused(client, "CAsuspended", NUMBER)
    assert client.portal.call(client.app.state.services.allocator.occupied) == 0


def test_route_revocation_stops_admission(client):  # noqa: F811
    from tests.test_inbound import refused

    publish(client)
    response = client.delete(f"/inbound/routes/{NUMBER}?client_id=client-1", headers=ADMIN)
    assert response.status_code == 200
    refused(client, "CArevoked", NUMBER)
    assert client.portal.call(client.app.state.services.allocator.occupied) == 0


def test_revocation_fences_a_delayed_publish_even_after_inactive_transfer(client):  # noqa: F811
    from tests.test_inbound import route_body

    publish(client)
    assert (
        client.delete(f"/inbound/routes/{NUMBER}?client_id=client-1&minimum_version=20", headers=ADMIN).status_code
        == 200
    )
    assert publish(client, version=1).status_code == 409
    client.put("/tenants", json={"id": "client-2", "name": "Next", "status": "active"}, headers=ADMIN)
    body = route_body(clientId="client-2", version=21, active=False)
    assert client.put("/inbound/routes", json=body, headers=ADMIN).status_code == 200
    assert publish(client, version=1).status_code == 409
