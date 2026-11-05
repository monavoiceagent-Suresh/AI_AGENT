from tests.test_api import ADMIN, client  # noqa: F401
from tests.test_fairness import insert_job_with_leads


def test_capacity_uses_one_ledger_snapshot_and_only_eligible_queue(client):  # noqa: F811
    ids = insert_job_with_leads(client, "tenant-a", "capacity-a", ["+919876502001", "+919876502002"])
    pool = client.app.state.services.pool

    async def seed():
        await pool.execute("UPDATE dispatch_leads SET attempts = 5 WHERE id = $1", ids[1])
        await pool.execute(
            "INSERT INTO channel_allocations (id, direction, state, client_id) VALUES "
            "('a', 'outbound', 'connected', 'tenant-a'), "
            "('b', 'outbound', 'releasing', 'tenant-a'), "
            "('c', 'inbound', 'reserved', NULL)"
        )

    client.portal.call(seed)
    snapshot = client.get("/capacity", headers=ADMIN).json()
    assert snapshot["occupied"] == snapshot["connected"] + snapshot["reserved"] == 3
    assert snapshot["outboundOccupied"] == 2 and snapshot["inboundOccupied"] == 1
    assert snapshot["unattributed"] == 1
    assert snapshot["tenants"] == [{"clientId": "tenant-a", "connected": 1, "reserved": 1, "queuedLeads": 1}]

    async def close_window():
        await pool.execute("UPDATE dispatch_jobs SET window_start = '00:00', window_end = '00:00'")

    client.portal.call(close_window)
    assert client.get("/capacity", headers=ADMIN).json()["tenants"][0]["queuedLeads"] == 0
