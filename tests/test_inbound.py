"""Inbound routing: publishing routes, answering by the dialled number, isolation and reserved capacity."""

import json
import logging

import asyncpg
import pytest
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from app.phone import normalize_phone
from tests.fakes import exotel_start, make_settings
from tests.test_api import ADMIN, APPOINTMENT_AGENT, EXOTEL_AUTH, client, converse  # noqa: F401
from tests.test_dispatch import dispatch_body, ready, sink, start, wait_until  # noqa: F401

NUMBER = "+918047491899"
OTHER_NUMBER = "+919999900001"
PERSONA = {
    "key": "front-desk",
    "name": "Front desk",
    "language": "en",
    "workflow": "appointment",
    "instructions": "You are the front desk of Sunrise Clinic. Context: [[CALLER_CONTEXT]]",
    "greeting": "Welcome to Sunrise Clinic.",
    "knowledge": "",
    "version": 1,
}


def route_body(**overrides) -> dict:
    body = {"phone": NUMBER, "clientId": "client-1", "version": 1, "active": True, "persona": PERSONA}
    return body | overrides


def publish(client, **overrides):  # noqa: F811
    body = route_body(**overrides)
    client.put("/tenants", json={"id": body["clientId"], "name": "Test", "status": "active"}, headers=ADMIN)
    return client.put("/inbound/routes", json=body, headers=ADMIN)


def refused(client, call_sid: str, to_number: str) -> None:  # noqa: F811
    """The engine closes the stream without answering."""
    with client.websocket_connect("/exotel/stream", headers=EXOTEL_AUTH) as ws:
        ws.send_text(json.dumps({"event": "connected"}))
        ws.send_text(exotel_start(call_sid, to_number=to_number))
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def system_prompts(client) -> list[str]:  # noqa: F811
    return [request[0]["content"] for request in client.app.state.services.llm.requests]


def another_reply(client) -> None:  # noqa: F811
    client.app.state.services.llm.replies.append("Thank you, goodbye. <end_call>")


# ---- number format ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("+918047491899", "918047491899"),
        ("918047491899", "918047491899"),
        ("08047491899", "918047491899"),
        ("8047491899", "918047491899"),
        ("+91 80474-91899", "918047491899"),
        ("+14155550123", "14155550123"),
        ("", None),
        (None, None),
        ("abc", None),
        ("12345", None),
        ("+0918047491899", None),  # a country code cannot start with 0
        ("+9109513886363", "919513886363"),  # +91 followed by a trunk 0
        ("+91 0 95138 86363", "919513886363"),
    ],
)
def test_the_dialled_number_is_normalised_however_exotel_writes_it(raw, expected):
    assert normalize_phone(raw) == expected


# ---- publishing ------------------------------------------------------------------------------


def test_publishing_needs_the_token_and_a_valid_route(client):  # noqa: F811
    assert client.put("/inbound/routes", json=route_body()).status_code == 401
    assert publish(client, phone="8047491899").status_code == 422  # E.164 with + only
    assert publish(client, version=0).status_code == 422
    assert publish(client, persona=PERSONA | {"greeting": ""}).status_code == 422
    assert publish(client, persona=PERSONA | {"workflow": "support"}).status_code == 422
    assert client.get("/inbound/routes/8047491899").status_code == 401


def test_a_published_route_can_be_read_back(client):  # noqa: F811
    assert client.get(f"/inbound/routes/{NUMBER}", headers=ADMIN).status_code == 404
    applied = publish(client)
    assert applied.status_code == 200 and applied.json()["status"] == "applied"
    route = client.get("/inbound/routes/08047491899", headers=ADMIN).json()  # any way of writing the number
    assert route == {
        "phone": "918047491899",
        "clientId": "client-1",
        "version": 1,
        "active": True,
        "personaKey": "front-desk",
        "personaVersion": 1,
        "workflow": "appointment",
        "status": "current",
    }


def test_versions_only_move_forward_and_a_replay_is_harmless(client):  # noqa: F811
    assert publish(client).json()["status"] == "applied"
    assert publish(client).json()["status"] == "unchanged"

    edited = PERSONA | {"greeting": "Different greeting."}
    conflict = publish(client, persona=edited)  # same version, different content
    assert conflict.status_code == 409 and conflict.json()["detail"] == {"code": "version_conflict"}

    assert publish(client, version=2, persona=edited).json()["status"] == "applied"
    stale = publish(client, version=1)
    assert stale.status_code == 409
    assert stale.json()["detail"] == {"code": "stale_version", "currentVersion": 2}
    assert client.get(f"/inbound/routes/{NUMBER}", headers=ADMIN).json()["version"] == 2


def test_a_number_never_moves_to_another_workspace(client):  # noqa: F811
    assert publish(client).status_code == 200
    taken = publish(client, clientId="client-2", version=9, persona=PERSONA | {"instructions": "Not yours."})
    assert taken.status_code == 409 and taken.json()["detail"] == {"code": "number_taken"}
    route = client.get(f"/inbound/routes/{NUMBER}", headers=ADMIN).json()
    assert (route["clientId"], route["version"]) == ("client-1", 1)


def test_a_persona_the_engine_cannot_speak_is_refused(client):  # noqa: F811
    client.app.state.services.settings.bakbak_voices.pop("te")
    refusal = publish(client, persona=PERSONA | {"language": "te"})
    assert refusal.status_code == 422 and refusal.json()["detail"]["code"] == "persona_unspeakable"
    assert client.get(f"/inbound/routes/{NUMBER}", headers=ADMIN).status_code == 404


def test_each_workspace_keeps_its_own_agent_row_for_a_persona_key(client):  # noqa: F811
    publish(client)
    publish(client, phone=OTHER_NUMBER, clientId="client-2", persona=PERSONA | {"name": "Other workspace"})
    agents = client.get("/agents", headers=ADMIN).json()
    assert sorted((a["client_id"], a["persona_key"]) for a in agents) == [
        ("client-1", "front-desk"),
        ("client-2", "front-desk"),
    ]


# ---- answering by the dialled number ---------------------------------------------------------


def test_a_routed_number_is_answered_with_its_published_persona(client):  # noqa: F811
    publish(client)
    converse(client, "CAin1", to_number="08047491899")  # Exotel may write the number without +91

    assert "You are the front desk of Sunrise Clinic." in system_prompts(client)[0]
    assert "Inbound call" in system_prompts(client)[0] and "[[" not in system_prompts(client)[0]
    assert client.app.state.services.tts.texts[0] == "Welcome to Sunrise Clinic."
    call = wait_until(lambda: next((c for c in client.get("/calls", headers=ADMIN).json() if c["outcome"]), None))
    assert call["direction"] == "inbound" and call["outcome"] == "appointment_requested"
    assert call["to_number"] == "08047491899"


def test_a_republished_route_applies_to_the_next_call_only(client):  # noqa: F811
    publish(client)
    converse(client, "CAin1", to_number=NUMBER)
    publish(client, version=2, persona=PERSONA | {"instructions": "Second text. Context: [[CALLER_CONTEXT]]"})
    another_reply(client)
    converse(client, "CAin2", to_number=NUMBER)
    first, second = system_prompts(client)
    assert "front desk of Sunrise Clinic" in first and "Second text." in second


def test_numbers_of_different_workspaces_never_share_a_persona(client):  # noqa: F811
    publish(client, persona=PERSONA | {"instructions": "ALPHA workspace text. Context: [[CALLER_CONTEXT]]"})
    publish(
        client,
        phone=OTHER_NUMBER,
        clientId="client-2",
        persona=PERSONA | {"instructions": "BETA workspace text. Context: [[CALLER_CONTEXT]]"},
    )
    another_reply(client)
    converse(client, "CAin1", to_number=NUMBER)
    converse(client, "CAin2", to_number=OTHER_NUMBER)
    alpha, beta = system_prompts(client)
    assert "ALPHA" in alpha and "BETA" not in alpha
    assert "BETA" in beta and "ALPHA" not in beta


def test_an_unrouted_number_is_refused_when_the_default_agent_is_off(client):  # noqa: F811
    client.post("/agents", json=APPOINTMENT_AGENT, headers=ADMIN)  # an engine-wide default exists
    publish(client)
    client.app.state.services.settings.inbound_legacy_default_enabled = False

    refused(client, "CAin1", to_number="+919000000000")
    refused(client, "CAin2", to_number="")  # a start event with no usable number
    assert client.get("/calls", headers=ADMIN).json() == []
    assert system_prompts(client) == []


def test_an_unrouted_number_still_gets_the_default_agent_by_default(client):  # noqa: F811
    client.post("/agents", json=APPOINTMENT_AGENT, headers=ADMIN)
    publish(client)
    converse(client, "CAin1", to_number="+919000000000")
    assert "Sunrise Clinic" in system_prompts(client)[0]
    assert client.app.state.services.tts.texts[0] == "Hello from Sunrise Clinic."  # the default agent, not the route


def test_a_workspaces_agent_cannot_be_flagged_as_the_default(client):  # noqa: F811
    """Flagging an agent default is not a way of choosing a persona for a number: it is refused, loudly."""
    owned = APPOINTMENT_AGENT | {"client_id": "client-1", "persona_key": "front-desk"}  # is_inbound_default: True
    refusal = client.post("/agents", json=owned, headers=ADMIN)
    assert refusal.status_code == 422 and "publish an inbound route" in refusal.text

    created = client.post("/agents", json=owned | {"is_inbound_default": False}, headers=ADMIN).json()
    via_update = client.put(f"/agents/{created['id']}", json=owned, headers=ADMIN)
    assert via_update.status_code == 422

    pool = client.app.state.services.pool
    with pytest.raises(asyncpg.CheckViolationError, match="agents_default_belongs_to_no_workspace"):
        client.portal.call(pool.execute, "UPDATE agents SET is_inbound_default = true WHERE id = $1", created["id"])
    refused(client, "CAin1", to_number="+919000000000")  # and nothing answers an unrouted number
    assert system_prompts(client) == []


def test_a_switched_off_route_is_refused(client):  # noqa: F811
    publish(client)
    publish(client, version=2, active=False)
    refused(client, "CAin1", to_number=NUMBER)
    publish(client, version=3, active=True)
    converse(client, "CAin2", to_number=NUMBER)
    assert len(system_prompts(client)) == 1


def test_inbound_can_be_disabled_entirely(client):  # noqa: F811
    publish(client)
    client.app.state.services.settings.enable_inbound_calls = False
    refused(client, "CAin1", to_number=NUMBER)


# ---- reserved capacity -----------------------------------------------------------------------
# Unit coverage of the admission logic itself (the inbound-reserved carve-out, concurrency,
# duplicate reservations, stale-reservation sweep) now lives in tests/test_allocator.py, against
# the real M2 ChannelAllocator. The tests below exercise the same business rule end to end,
# through the actual API and dispatcher.


def test_the_reservation_cannot_exceed_the_limit():
    with pytest.raises(ValidationError, match="INBOUND_RESERVED_CALLS"):
        make_settings(max_concurrent_calls=1, inbound_reserved_calls=2)


def test_a_manual_outbound_call_cannot_take_the_reserved_slot(client):  # noqa: F811
    publish(client)
    settings = client.app.state.services.settings
    settings.enable_outbound_calls, settings.max_concurrent_calls, settings.inbound_reserved_calls = True, 2, 1
    agent = client.post("/agents", json=APPOINTMENT_AGENT, headers=ADMIN).json()
    body = {"agent_id": agent["id"], "to": "+919876543210"}

    assert client.post("/calls", json=body, headers=ADMIN).status_code == 201
    assert client.post("/calls", json=body, headers=ADMIN).status_code == 409  # the second slot is for inbound
    converse(client, "CAin1", to_number=NUMBER)  # and an inbound call still gets through
    assert system_prompts(client)


def test_a_campaign_cannot_take_the_reserved_slot(ready, sink):  # noqa: F811
    publish(ready)
    settings = ready.app.state.services.settings
    settings.max_concurrent_calls, settings.inbound_reserved_calls = 2, 1
    leads = [{"id": f"lead-{n}", "phone": f"+91987654321{n}", "name": "A"} for n in range(2)]
    start(ready, leads=leads)

    wait_until(lambda: ready.app.state.services.exotel.dialled)
    wait_until(lambda: len(ready.get("/calls", headers=ADMIN).json()) == 1)
    assert len(ready.app.state.services.exotel.dialled) == 1  # the second lead waits: one slot is reserved
    converse(ready, "CAin1", to_number=NUMBER)
    assert "front desk of Sunrise Clinic" in system_prompts(ready)[0]


def test_inbound_is_refused_cleanly_when_every_slot_is_taken(client):  # noqa: F811
    publish(client)
    client.app.state.services.settings.enable_outbound_calls = True  # one slot in total
    agent = client.post("/agents", json=APPOINTMENT_AGENT, headers=ADMIN).json()
    body = {"agent_id": agent["id"], "to": "+919876543210"}
    assert client.post("/calls", json=body, headers=ADMIN).status_code == 201  # rings, holding the only slot

    refused(client, "CAin1", to_number=NUMBER)
    assert [c["direction"] for c in client.get("/calls", headers=ADMIN).json()] == ["outbound"]
    assert system_prompts(client) == []


def test_the_slot_taken_while_an_inbound_call_is_set_up_is_released(client):  # noqa: F811
    publish(client)
    converse(client, "CAin1", to_number=NUMBER)
    services = client.app.state.services
    assert client.portal.call(services.allocator.occupied) == 0


# ---- a route that does not match the dialled number (the reported bug) ----------------------------


EXOPHONE = "08037091588"  # how Exotel really reports the number that was dialled


def test_a_route_for_another_number_never_answers_and_the_call_is_visible_as_unrouted(client, caplog):  # noqa: F811
    """A route keyed to a different number does not answer the call. The default agent does, and the log says so."""
    caplog.set_level(logging.INFO)
    client.post("/agents", json=APPOINTMENT_AGENT, headers=ADMIN)  # the engine-wide default agent
    publish(client, phone="+919513886363")  # published for a number that is not the one Exotel dials

    converse(client, "CAin1", to_number=EXOPHONE)
    assert "Sunrise Clinic" in system_prompts(client)[0]
    assert client.app.state.services.tts.texts[0] == "Hello from Sunrise Clinic."  # the default agent, not the route
    assert "matches no published route" in caplog.text and "engine-wide default agent" in caplog.text
    assert "'08037091588' (read as 918037091588)" in caplog.text

    unrouted = client.get("/inbound/unrouted", headers=ADMIN).json()
    assert [(u["phone"], u["rawExample"], u["calls"], u["answeredBy"]) for u in unrouted] == [
        ("918037091588", EXOPHONE, 1, "default_agent")
    ]


def test_publishing_the_dialled_number_makes_the_persona_answer_and_clears_the_warning(client, caplog):  # noqa: F811
    caplog.set_level(logging.INFO)
    client.post("/agents", json=APPOINTMENT_AGENT, headers=ADMIN)
    converse(client, "CAin1", to_number=EXOPHONE)
    assert client.get("/inbound/unrouted", headers=ADMIN).json()[0]["calls"] == 1

    assert publish(client, phone="+918037091588").status_code == 200
    assert client.get("/inbound/unrouted", headers=ADMIN).json() == []  # a routed number is no longer a problem

    another_reply(client)
    caplog.clear()
    converse(client, "CAin2", to_number=EXOPHONE)
    assert "You are the front desk of Sunrise Clinic." in system_prompts(client)[1]
    assert client.app.state.services.tts.texts[-2] == "Welcome to Sunrise Clinic."  # the route's greeting
    agent = next(a for a in client.get("/agents", headers=ADMIN).json() if a["persona_key"] == "front-desk")
    assert f"on agent {agent['id']}" in caplog.text and "persona 'front-desk' (rev 1)" in caplog.text
    route = client.get("/inbound/routes/+918037091588", headers=ADMIN).json()
    assert (route["version"], route["personaKey"], route["clientId"]) == (1, "front-desk", "client-1")


def test_a_refused_unrouted_call_is_recorded_too(client):  # noqa: F811
    client.app.state.services.settings.inbound_legacy_default_enabled = False
    refused(client, "CAin1", to_number=EXOPHONE)
    refused(client, "CAin2", to_number="+918037091588")  # the same number written another way counts together
    unrouted = client.get("/inbound/unrouted", headers=ADMIN).json()
    assert [(u["phone"], u["calls"], u["answeredBy"]) for u in unrouted] == [("918037091588", 2, "refused")]
    assert client.get("/inbound/unrouted").status_code == 401  # authenticated like the rest of the management API


def test_a_malformed_indian_number_is_refused_when_publishing(client):  # noqa: F811
    typo = publish(client, phone="+9109513886363")  # +91 and then a national number with its trunk 0
    assert typo.status_code == 422
    assert "did you mean +919513886363" in typo.text
    assert publish(client, phone="+91951388636").status_code == 422  # one digit short
    assert publish(client, phone="+14155550123").status_code == 200  # other countries are not second-guessed
    assert client.get("/inbound/routes/+9109513886363", headers=ADMIN).status_code == 404


def test_the_saved_route_carries_the_agent_persona_and_workspace_it_was_published_with(client):  # noqa: F811
    publish(client, persona=PERSONA | {"version": 4, "greeting": "Latest greeting.", "knowledge": "Open nine to six."})
    pool = client.app.state.services.pool
    row = client.portal.call(
        pool.fetchrow,
        "SELECT r.client_id, r.version, r.active, r.persona, a.client_id AS agent_client, a.persona_key "
        "FROM inbound_routes r JOIN agents a ON a.id = r.agent_id WHERE r.phone = '918047491899'",
    )
    assert (row["client_id"], row["agent_client"], row["persona_key"]) == ("client-1", "client-1", "front-desk")
    assert row["persona"] == PERSONA | {
        "version": 4,
        "greeting": "Latest greeting.",
        "knowledge": "Open nine to six.",
        "outcome_rules": [],
        "outcome_fallback": None,
        "asr_provider": None,
        "asr_fallback_provider": None,
        "tts_provider": None,
        "tts_fallback_provider": None,
    }
