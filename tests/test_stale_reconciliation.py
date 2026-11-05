"""Stale reconciliation requires positive provider termination evidence."""

from app.providers.exotel import ExotelCallRecord
from tests.test_api import client  # noqa: F401
from tests.test_fairness import insert_job_with_leads  # noqa: F401


def seed_stale_call(client, call_sid: str, lead_pk: int):  # noqa: F811
    """One `calls` row, 'initiated' long enough ago to be a stale-ringing candidate, linked to a
    real dispatch lead so candidate_stale_calls' JOIN conditions are met."""
    pool = client.app.state.services.pool

    async def go():
        await client.app.state.services.allocator.reserve(call_sid, "outbound", call_sid=call_sid)
        return await pool.fetchval(
            "INSERT INTO calls (agent_id, direction, exotel_call_sid, status, created_at, "
            "dispatch_lead_id) SELECT j.agent_id, 'outbound', $1, 'initiated', "
            "now() - interval '10 minutes', $2 FROM dispatch_leads l JOIN dispatch_jobs j "
            "ON j.id = l.job_id WHERE l.id = $2 RETURNING id",
            call_sid,
            lead_pk,
        )

    return client.portal.call(go)


def test_a_call_exotel_still_reports_in_progress_is_not_abandoned(client):  # noqa: F811
    lead_ids = insert_job_with_leads(client, "tenant-a", "campaign-stale-1", ["+919876500010"])
    call_id = seed_stale_call(client, "CAstale1", lead_ids[0])
    svc = client.app.state.services
    svc.exotel.records["CAstale1"] = ExotelCallRecord(
        to="", phone_number_sid="", direction="outbound", forwarded_from="", status="in-progress"
    )

    from app import dispatch

    client.portal.call(dispatch.abandon_stale_calls, svc)

    status = client.portal.call(svc.pool.fetchval, "SELECT status FROM calls WHERE id = $1", call_id)
    assert status == "initiated"  # left alone: still genuinely connected per Exotel
    assert client.portal.call(svc.allocator.occupied) == 1  # its channel was not released either


def test_a_call_exotel_confirms_ended_is_abandoned_and_its_channel_released(client):  # noqa: F811
    lead_ids = insert_job_with_leads(client, "tenant-a", "campaign-stale-2", ["+919876500011"])
    call_id = seed_stale_call(client, "CAstale2", lead_ids[0])
    svc = client.app.state.services
    svc.exotel.records["CAstale2"] = ExotelCallRecord(
        to="", phone_number_sid="", direction="outbound", forwarded_from="", status="completed"
    )

    from app import dispatch

    client.portal.call(dispatch.abandon_stale_calls, svc)

    status = client.portal.call(svc.pool.fetchval, "SELECT status FROM calls WHERE id = $1", call_id)
    assert status == "abandoned"
    assert client.portal.call(svc.allocator.occupied) == 0


def test_an_unconfirmable_call_retains_its_channel(client):  # noqa: F811
    """Uncertainty is not proof of termination."""
    lead_ids = insert_job_with_leads(client, "tenant-a", "campaign-stale-3", ["+919876500012"])
    call_id = seed_stale_call(client, "CAstale3", lead_ids[0])
    svc = client.app.state.services
    svc.exotel.fail_lookups = 99

    from app import dispatch

    client.portal.call(dispatch.abandon_stale_calls, svc)

    status = client.portal.call(svc.pool.fetchval, "SELECT status FROM calls WHERE id = $1", call_id)
    assert status == "initiated"
    assert client.portal.call(svc.allocator.occupied) == 1
