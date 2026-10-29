"""The number an inbound call is routed by.

Facts these tests encode, from Exotel's own call records for the account's inbound calls: the stream's `to` is
Exotel's record `To` (08037091588), which is not the ExoPhone the caller dialled (09513886363, Exotel's
`PhoneNumberSid`, also the account's caller ID). Exotel documents the shape of the `start` frame but not what
`from` and `to` mean.
"""

import json
import logging

import httpx
import pytest

from app import calls, reporting
from app.phone import normalize_phone
from app.providers.exotel import ExotelCallRecord, ExotelClient, ExotelError
from app.redaction import mask_number, redact
from tests.test_api import ADMIN, APPOINTMENT_AGENT, EXOTEL_AUTH, client, converse  # noqa: F401
from tests.test_dispatch import wait_until
from tests.test_inbound import PERSONA, another_reply, publish, refused, system_prompts
from tests.test_reporting import mailbox  # noqa: F401

STREAM_TO = "08037091588"  # what Exotel puts in the stream
EXOPHONE = "09513886363"  # the number dialled (Exotel's PhoneNumberSid)
CALLER = "09360113501"
ENQUIRY = PERSONA | {"instructions": "ENQUIRY PERSONA text. Context: [[CALLER_CONTEXT]]", "greeting": "Enquiry hello."}


def record(client, call_sid: str, **over):  # noqa: F811
    fields = {"to": STREAM_TO, "phone_number_sid": EXOPHONE, "direction": "inbound", "forwarded_from": ""} | over
    client.app.state.services.exotel.records[call_sid] = ExotelCallRecord(**fields)


def stored_dialled(client) -> list[str | None]:  # noqa: F811
    pool = client.app.state.services.pool
    rows = client.portal.call(pool.fetch, "SELECT dialled_number FROM calls ORDER BY id")
    return [r["dialled_number"] for r in rows]


def lookups(client) -> list[str]:  # noqa: F811
    return client.app.state.services.exotel.lookups


# ---- numbers -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (STREAM_TO, "918037091588"),
        (EXOPHONE, "919513886363"),
        ("+9109513886363", "919513886363"),  # the way it was saved from the dashboard before it was validated
        ("919513886363", "919513886363"),
    ],
)
def test_the_real_numbers_normalise_as_expected(raw, expected):
    assert normalize_phone(raw) == expected


# ---- routing by the dialled ExoPhone --------------------------------------------------------------


def test_a_route_for_the_dialled_exophone_answers_when_the_stream_to_is_another_number(client, mailbox, caplog):  # noqa: F811
    caplog.set_level(logging.INFO)
    publish(client, phone="+919513886363", persona=ENQUIRY)
    record(client, "CAin1")

    converse(client, "CAin1", to_number=STREAM_TO, from_number=CALLER)

    assert "ENQUIRY PERSONA text." in system_prompts(client)[0]
    assert client.app.state.services.tts.texts[0] == "Enquiry hello."
    assert lookups(client) == ["CAin1"]
    assert stored_dialled(client) == ["919513886363"]
    assert "found by the ExoPhone in Exotel's call record" in caplog.text
    wait_until(lambda: mailbox.deliveries)  # the report carries the routed number so the dashboard can find its owner
    assert json.loads(mailbox.deliveries[0]["body"])["data"]["dialledNumber"] == "919513886363"
    call = client.get("/calls", headers=ADMIN).json()[0]
    assert call["to_number"] == STREAM_TO  # what Exotel sent is still stored exactly as received


def test_the_reported_case_a_route_saved_with_a_stray_trunk_zero_still_answers(client, caplog):  # noqa: F811
    """The route in the user's database is keyed 9109513886363 (+91 then 0 9513886363); calls arrive as 08037091588."""
    caplog.set_level(logging.INFO)
    agent = client.post(
        "/agents",
        json=APPOINTMENT_AGENT | {"is_inbound_default": False, "client_id": "client-1", "persona_key": "front-desk"},
        headers=ADMIN,
    ).json()
    pool = client.app.state.services.pool
    client.portal.call(
        pool.execute,
        "INSERT INTO inbound_routes (phone, client_id, agent_id, version, active, persona) "
        "VALUES ('9109513886363', 'client-1', $1, 3, true, $2)",
        agent["id"],
        ENQUIRY,
    )
    record(client, "CAin1")

    converse(client, "CAin1", to_number=STREAM_TO, from_number=CALLER)

    assert "ENQUIRY PERSONA text." in system_prompts(client)[0]
    assert stored_dialled(client) == ["9109513886363"]  # the key that matched, so the dashboard's mapping lines up
    assert f"on agent {agent['id']}" in caplog.text
    # Looking a number up for a call is tolerant; reading a route back is exact.
    assert client.get("/inbound/routes/+919513886363", headers=ADMIN).status_code == 404


def test_a_route_for_the_stream_to_wins_and_no_lookup_is_made(client):  # noqa: F811
    publish(client, phone="+918037091588", persona=ENQUIRY)
    record(client, "CAin1", phone_number_sid="09000000000")

    converse(client, "CAin1", to_number=STREAM_TO)

    assert "ENQUIRY PERSONA text." in system_prompts(client)[0]
    assert lookups(client) == []
    assert stored_dialled(client) == ["918037091588"]


def test_the_number_that_matches_is_the_one_in_the_stream_when_both_have_routes(client):  # noqa: F811
    publish(client, phone="+918037091588", persona=ENQUIRY)
    publish(client, phone="+919513886363", persona=PERSONA | {"instructions": "OTHER text. [[CALLER_CONTEXT]]"})
    record(client, "CAin1")
    converse(client, "CAin1", to_number=STREAM_TO)
    assert "ENQUIRY PERSONA text." in system_prompts(client)[0]
    assert "OTHER text." not in system_prompts(client)[0]


# ---- the no-route safeguard is kept ---------------------------------------------------------------


def test_a_number_nothing_routes_is_refused_and_recorded_under_the_dialled_exophone(client, caplog):  # noqa: F811
    caplog.set_level(logging.INFO)
    publish(client, phone="+919513886363", persona=ENQUIRY)  # a route exists, for another number
    client.app.state.services.settings.inbound_legacy_default_enabled = False
    record(client, "CAin1", phone_number_sid="09000000000")

    refused(client, "CAin1", to_number=STREAM_TO)

    assert system_prompts(client) == []  # no persona answered
    unrouted = client.get("/inbound/unrouted", headers=ADMIN).json()
    assert [(u["phone"], u["answeredBy"]) for u in unrouted] == [("919000000000", "refused")]
    assert "08037091588" in unrouted[0]["rawExample"] and "919000000000" in unrouted[0]["rawExample"]
    assert "matches no published route" in caplog.text and "Exotel ExoPhone 919000000000" in caplog.text


def test_the_global_default_still_answers_unrouted_numbers_only_when_enabled(client):  # noqa: F811
    client.post("/agents", json=APPOINTMENT_AGENT, headers=ADMIN)  # the engine-wide default
    publish(client, phone="+919513886363", persona=ENQUIRY)
    record(client, "CAin1", phone_number_sid="09000000000")
    converse(client, "CAin1", to_number=STREAM_TO)
    assert "ENQUIRY PERSONA text." not in system_prompts(client)[0]  # the default agent, never the workspace persona
    assert client.app.state.services.tts.texts[0] == "Hello from Sunrise Clinic."


def test_a_failed_lookup_refuses_the_call_safely(client, monkeypatch):  # noqa: F811
    monkeypatch.setattr(calls, "LOOKUP_RETRY_SECONDS", 0)
    publish(client, phone="+919513886363", persona=ENQUIRY)
    client.app.state.services.settings.inbound_legacy_default_enabled = False
    # no Exotel record for this call: the fake raises ExotelError(404), as the real client does for a missing record
    refused(client, "CAin1", to_number=STREAM_TO)
    assert lookups(client) == ["CAin1", "CAin1"] and system_prompts(client) == []
    unrouted = client.get("/inbound/unrouted", headers=ADMIN).json()
    assert [(u["phone"], u["rawExample"]) for u in unrouted] == [("918037091588", STREAM_TO)]


def test_a_call_exotel_has_not_recorded_yet_is_read_again_once(client, monkeypatch):  # noqa: F811
    monkeypatch.setattr(calls, "LOOKUP_RETRY_SECONDS", 0)
    publish(client, phone="+919513886363", persona=ENQUIRY)
    record(client, "CAin1")
    client.app.state.services.exotel.fail_lookups = 1

    converse(client, "CAin1", to_number=STREAM_TO)

    assert "ENQUIRY PERSONA text." in system_prompts(client)[0]
    assert lookups(client) == ["CAin1", "CAin1"]


def test_the_lookup_gives_up_after_one_retry(client, monkeypatch):  # noqa: F811
    monkeypatch.setattr(calls, "LOOKUP_RETRY_SECONDS", 0)
    publish(client, phone="+919513886363", persona=ENQUIRY)
    client.app.state.services.settings.inbound_legacy_default_enabled = False
    record(client, "CAin1")
    client.app.state.services.exotel.fail_lookups = 5
    refused(client, "CAin1", to_number=STREAM_TO)
    assert lookups(client) == ["CAin1", "CAin1"] and system_prompts(client) == []


def test_the_lookup_can_be_turned_off(client):  # noqa: F811
    publish(client, phone="+919513886363", persona=ENQUIRY)
    client.app.state.services.settings.inbound_legacy_default_enabled = False
    client.app.state.services.settings.exotel_resolve_called_number = False
    record(client, "CAin1")
    refused(client, "CAin1", to_number=STREAM_TO)
    assert lookups(client) == []


@pytest.mark.parametrize(
    "bad",
    [
        {"direction": "outbound-api"},  # not an inbound call
        {"phone_number_sid": "a1b2c3d4e5f60718293a4b5c6d7e8f90"},  # a SID, not a number
        {"phone_number_sid": ""},
    ],
)
def test_an_exotel_record_that_does_not_name_a_dialled_number_is_ignored(client, bad):  # noqa: F811
    publish(client, phone="+919513886363", persona=ENQUIRY)
    client.app.state.services.settings.inbound_legacy_default_enabled = False
    record(client, "CAin1", **bad)
    refused(client, "CAin1", to_number=STREAM_TO)
    assert system_prompts(client) == []


def test_a_workspace_is_never_answered_with_another_workspaces_persona(client):  # noqa: F811
    publish(
        client,
        phone="+918037091588",
        clientId="client-1",
        persona=ENQUIRY | {"instructions": "ALPHA text. [[CALLER_CONTEXT]]"},
    )
    publish(
        client,
        phone="+919513886363",
        clientId="client-2",
        persona=ENQUIRY | {"instructions": "BETA text. [[CALLER_CONTEXT]]"},
    )
    record(client, "CAin1", phone_number_sid="09513886363")
    # the stream's own `to` matches client-1's number, so client-1's persona answers and the lookup is never needed
    converse(client, "CAin1", to_number=STREAM_TO)
    another_reply(client)
    record(client, "CAin2", to="08044445555", phone_number_sid="09513886363")
    converse(client, "CAin2", to_number="08044445555")  # unmatched stream `to`: the ExoPhone is client-2's number
    alpha, beta = system_prompts(client)
    assert "ALPHA text." in alpha and "BETA text." not in alpha
    assert "BETA text." in beta and "ALPHA text." not in beta


# ---- diagnostics ----------------------------------------------------------------------------------


def test_the_stream_start_is_logged_for_diagnosis_without_credentials_or_callers(client, caplog):  # noqa: F811
    caplog.set_level(logging.INFO)
    publish(client, phone="+918037091588", persona=ENQUIRY)
    converse(
        client,
        "CAin1",
        to_number=STREAM_TO,
        from_number=CALLER,
        extra={"forwarded_from": "+919999912345", "exophone": "09513886363"},
        custom_parameters={"queue": "premium", "api_key": "SECRET-VALUE", "callback": "https://user:pw@example.test/x"},
    )

    start_line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("exotel stream start "))
    shown = json.loads(start_line.removeprefix("exotel stream start "))
    assert shown["start_keys"] == sorted(
        [
            "stream_sid",
            "call_sid",
            "account_sid",
            "from",
            "to",
            "custom_parameters",
            "media_format",
            "forwarded_from",
            "exophone",
        ]
    )
    assert shown["events_before_start"] == [{"event": "connected", "keys": ["event"]}]
    assert shown["start"]["to"] == STREAM_TO  # the business-side number is shown in full
    assert shown["start"]["call_sid"] == "CAin1"
    assert shown["start"]["from"] == "*******3501"  # the caller only by its last digits
    assert shown["start"]["forwarded_from"] == "********2345"
    assert shown["start"]["exophone"] == "*******6363"  # an unexpected number field is still recognisable
    assert shown["start"]["account_sid"] == "...test"
    assert shown["start"]["custom_parameters"] == {
        "queue": "premium",
        "api_key": "[redacted]",
        "callback": "https://[redacted]@example.test/x",
    }

    request_line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("exotel stream request "))
    request = json.loads(request_line.removeprefix("exotel stream request "))
    assert request["path"] == "/exotel/stream" and request["authorization"] == "present"

    everything = caplog.text
    for secret in (
        "SECRET-VALUE",
        "user:pw",
        "test-token",
        EXOTEL_AUTH["Authorization"].split()[1],
        CALLER,
        "9360113501",
    ):
        assert secret not in everything


def test_redaction_rules():
    assert mask_number("+91 93601 13501") == "********3501"
    assert redact({"Authorization": "Basic abc", "token": "t", "n": 5, "ok": "hello"}) == {
        "Authorization": "[redacted]",
        "token": "[redacted]",
        "n": 5,
        "ok": "hello",
    }
    assert redact({"to": "08037091588", "from": "09360113501"}, keep_full=frozenset({"to"})) == {
        "to": "08037091588",
        "from": "*******3501",
    }
    deep = redact({"a": {"b": {"c": {"d": "deep"}}}})
    assert deep == {"a": {"b": {"c": "{...}"}}}
    assert len(redact("x" * 500)) == 63


# ---- Exotel's call record ------------------------------------------------------------------------


def exotel_client(handler):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ExotelClient(
        http,
        base_url="https://api.exotel.test/",
        account_sid="ACsid",
        api_key="key",
        api_token="token",
        caller_id=EXOPHONE,
        app_id="",
    )


@pytest.mark.anyio
async def test_the_call_record_is_read_with_the_accounts_credentials():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["auth"], seen["method"] = str(request.url), request.headers["authorization"], request.method
        return httpx.Response(
            200,
            json={"Call": {"To": STREAM_TO, "PhoneNumberSid": EXOPHONE, "Direction": "inbound", "ForwardedFrom": ""}},
        )

    call = await exotel_client(handler).get_call_record("CAxyz", wait_seconds=2)

    assert call == ExotelCallRecord(to=STREAM_TO, phone_number_sid=EXOPHONE, direction="inbound", forwarded_from="")
    assert seen["method"] == "GET"
    assert seen["url"] == "https://api.exotel.test/v1/Accounts/ACsid/Calls/CAxyz.json"
    assert seen["auth"].startswith("Basic ")


@pytest.mark.anyio
async def test_a_call_record_that_cannot_be_read_raises_a_plain_error():
    with pytest.raises(ExotelError, match=r"\(404\)"):
        await exotel_client(lambda request: httpx.Response(404, text="nope")).get_call_record("CAxyz", wait_seconds=2)
    empty = await exotel_client(lambda request: httpx.Response(200, json={})).get_call_record("CAxyz", wait_seconds=2)
    assert empty == ExotelCallRecord(to="", phone_number_sid="", direction="", forwarded_from="")


# ---- the report -----------------------------------------------------------------------------------


def test_the_report_carries_the_routed_number_and_falls_back_to_the_streams():
    base = {
        "id": 1,
        "from_number": CALLER,
        "to_number": STREAM_TO,
        "status": "completed",
        "connected_seconds": 3.0,
        "outcome": "resolved",
        "summary": "",
        "outcome_data": {},
    }
    routed = reporting.inbound_call_event(base | {"dialled_number": "919513886363"}, has_transcript=False)
    assert routed["data"]["dialledNumber"] == "919513886363"
    older = reporting.inbound_call_event(base | {"dialled_number": None}, has_transcript=False)
    assert older["data"]["dialledNumber"] == "918037091588"
