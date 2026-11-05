"""API and call-lifecycle tests against the docker-compose PostgreSQL (a throwaway database)."""

import asyncio
import base64
import json
import time
import uuid
from contextlib import asynccontextmanager

import asyncpg
import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.config import Settings, get_settings
from app.interfaces import CommittedTranscript
from app.providers.exotel import ExotelCallRecord, ExotelError
from tests.fakes import FakeASR, FakeLLM, FakeTTS, exotel_start

ADMIN = {"Authorization": "Bearer test-admin"}
EXOTEL_AUTH = {"Authorization": "Basic " + base64.b64encode(b"test-key:test-token").decode()}
APPOINTMENT_AGENT = {
    "name": "Clinic",
    "language": "en",
    "workflow": "appointment",
    "instructions": "You book appointments for Sunrise Clinic. Context: [[CALLER_CONTEXT]]",
    "greeting": "Hello from Sunrise Clinic.",
    "collect_fields": ["name", "preferred_time"],
    "is_inbound_default": True,
}


class TranscribingASR(FakeASR):
    """Hears one caller utterance as soon as caller audio arrives."""

    async def send_audio(self, pcm: bytes) -> None:
        await super().send_audio(pcm)
        if self.audio_bytes == len(pcm):
            self.queue.put_nowait(CommittedTranscript("I need an appointment tomorrow at ten"))


class FakeScribe:
    @asynccontextmanager
    async def connect(self, *, language: str, sample_rate: int):
        yield TranscribingASR()


class FakeExotel:
    def __init__(self) -> None:
        self.dialled: list[str] = []
        # Exotel's own call records by CallSid, and the lookups made, for the called-number resolution.
        self.records: dict[str, ExotelCallRecord] = {}
        self.lookups: list[str] = []
        self.fail_lookups = 0  # the first N lookups fail, like a call Exotel has not recorded yet

    async def connect_to_flow(self, number: str, **_: object) -> str:
        self.dialled.append(number)
        return f"CAout{len(self.dialled)}"

    async def get_call_record(self, call_sid: str, *, wait_seconds: float) -> ExotelCallRecord:
        self.lookups.append(call_sid)
        if self.fail_lookups > 0:
            self.fail_lookups -= 1
            raise ExotelError("Exotel call lookup failed (404)")
        if call_sid not in self.records:
            raise ExotelError("Exotel call lookup failed (404)")
        return self.records[call_sid]


async def _admin_dsn_and_db() -> tuple[str, str]:
    base = Settings().database_url
    from urllib.parse import urlsplit
    if urlsplit(base).hostname not in {"127.0.0.1", "localhost", "::1"}:
        pytest.skip("Integration fixtures require a loopback PostgreSQL host")
    name = f"voice_agent_test_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(base, timeout=5)
    try:
        await conn.execute(f"CREATE DATABASE {name}")
    finally:
        await conn.close()
    return base, name


async def _drop(base: str, name: str) -> None:
    conn = await asyncpg.connect(base)
    try:
        await conn.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")
    finally:
        await conn.close()


async def _check_redis(url: str) -> None:
    from redis.asyncio import from_url

    redis = from_url(url, socket_connect_timeout=2)
    try:
        await redis.ping()
    finally:
        await redis.aclose()


@pytest.fixture
def client(monkeypatch):
    try:
        base, name = asyncio.run(_admin_dsn_and_db())
    except (OSError, asyncpg.PostgresError) as exc:
        pytest.skip(f"PostgreSQL not reachable (docker compose up -d db): {exc}")
    redis_url = "redis://127.0.0.1:6379/0"
    try:
        asyncio.run(_check_redis(redis_url))
    except Exception as exc:
        pytest.skip(f"Redis not reachable (docker compose up -d redis): {exc}")
    test_dsn = base.rsplit("/", 1)[0] + f"/{name}"
    for key, value in {
        "DATABASE_URL": test_dsn,
        # A fresh prefix per test: this Redis instance is shared (unlike the throwaway Postgres
        # database above), so tests must not see each other's reservations.
        "REDIS_URL": redis_url,
        "REDIS_KEY_PREFIX": f"test-{uuid.uuid4().hex[:8]}",
        "APP_SECRET_KEY": "test-admin",
        "EXOTEL_API_KEY": "test-key",
        "EXOTEL_API_TOKEN": "test-token",
        "EXOTEL_WEBHOOK_SECRET": "test-hook",
        "ENABLE_INBOUND_CALLS": "true",
        "ENABLE_OUTBOUND_CALLS": "false",
        "MAX_CONCURRENT_CALLS": "1",
        "ELEVENLABS_TTS_MODEL_ID": "eleven_flash_v2_5",
        "TTS_PROVIDER": "bakbak",
        "TTS_FALLBACK_PROVIDER": "elevenlabs_standard",
        "BAKBAK_INR_PER_1K_CHARS": "2.5",
        "BAKBAK_API_KEY": "test-bakbak",
        "BAKBAK_VOICES": json.dumps(
            {lang: {"voice_id": f"{lang}-voice", "model_id": "m1"} for lang in ("en", "hi", "ta", "te")}
        ),
        "EXOTEL_INR_PER_MINUTE": "0.6",
        "DASHBOARD_WEBHOOK_SECRET": "test-dashboard-secret",
        "DISPATCHER_POLL_SECONDS": "0.05",
    }.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    from app.main import app

    try:
        with TestClient(app) as test_client:
            svc = app.state.services
            svc.asr = FakeScribe()
            svc.llm = FakeLLM(
                ["Sure, may I have your name? <end_call>"],
                outcome='{"outcome": "appointment_requested", "fields": {"name": null, "preferred_time": '
                '"tomorrow 10am"}, "summary": "Caller wants an appointment."}',
            )
            svc.tts = FakeTTS()
            svc.exotel = FakeExotel()
            yield test_client
    finally:
        get_settings.cache_clear()
        asyncio.run(_drop(base, name))


def test_management_api_requires_token(client):
    assert client.get("/agents").status_code == 401
    assert client.get("/agents", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/health").json()["status"] == "ok"


def test_agent_crud_and_validation(client):
    first = client.post("/agents", json=APPOINTMENT_AGENT, headers=ADMIN)
    assert first.status_code == 201, first.text
    second = client.post("/agents", json=APPOINTMENT_AGENT | {"name": "Clinic 2"}, headers=ADMIN).json()
    agents = client.get("/agents", headers=ADMIN).json()
    assert [a["is_inbound_default"] for a in agents] == [False, True]  # only one inbound default

    telugu = APPOINTMENT_AGENT | {"language": "te", "is_inbound_default": False}
    assert client.post("/agents", json=telugu, headers=ADMIN).status_code == 201  # default Telugu voice exists
    half_override = client.post("/agents", json=telugu | {"voice_id": "other-voice"}, headers=ADMIN)
    assert half_override.status_code == 422 and "together" in half_override.json()["detail"]
    bad_model = telugu | {"voice_id": "v", "tts_model_id": "eleven_v3"}
    assert client.post("/agents", json=bad_model, headers=ADMIN).status_code == 422
    both = telugu | {"voice_id": "v", "tts_model_id": "standard"}
    assert client.post("/agents", json=both, headers=ADMIN).status_code == 201

    updated = client.put(f"/agents/{second['id']}", json=APPOINTMENT_AGENT | {"name": "Renamed"}, headers=ADMIN)
    assert updated.json()["name"] == "Renamed"
    assert client.get("/agents/999", headers=ADMIN).status_code == 404


def test_voices_endpoint_requires_auth(client):
    assert client.get("/voices").status_code == 401


def test_voices_endpoint_proxies_the_configured_providers_real_voices(client):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers["x-api-key"]))
        return httpx.Response(200, json=[{"voice_id": "v1", "name": "Anitha", "model_id": "standard"}])

    client.app.state.services.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    response = client.get("/voices", headers=ADMIN)

    assert response.status_code == 200, response.text
    assert seen == [("https://hub.getraya.app/v1/voices", "test-bakbak")]  # the fixture's BAKBAK_API_KEY
    assert response.json() == {
        "provider": "bakbak",
        "voices": [{"voiceId": "v1", "name": "Anitha", "language": None, "modelId": "standard"}],
    }


def test_voices_endpoint_maps_a_provider_failure_to_502_not_a_crash(client):
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client.app.state.services.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    response = client.get("/voices", headers=ADMIN)

    assert response.status_code == 502


def test_outbound_call_lifecycle_until_status_callback(client):
    agent = client.post("/agents", json=APPOINTMENT_AGENT, headers=ADMIN).json()
    body = {"agent_id": agent["id"], "to": "+919876543210"}
    assert client.post("/calls", json=body, headers=ADMIN).status_code == 409  # outbound disabled

    client.app.state.services.settings.enable_outbound_calls = True
    placed = client.post("/calls", json=body, headers=ADMIN)
    assert placed.status_code == 201, placed.text
    assert placed.json()["status"] == "initiated" and placed.json()["exotel_call_sid"] == "CAout1"
    assert client.post("/calls", json=body, headers=ADMIN).status_code == 409  # capacity held while ringing

    assert client.post("/exotel/status?token=bad", data={"CallSid": "CAout1", "Status": "no-answer"}).status_code == 403
    ok = client.post("/exotel/status?token=test-hook", data={"CallSid": "CAout1", "Status": "no-answer"})
    assert ok.status_code == 200
    call = client.get(f"/calls/{placed.json()['id']}", headers=ADMIN).json()
    assert call["status"] == "no_answer" and call["ended_at"] is not None
    assert client.post("/calls", json=body, headers=ADMIN).status_code == 201  # capacity released


def _persona(**overrides) -> dict:
    return {
        "key": "clinic-test",
        "name": "Clinic",
        "language": "en",
        "workflow": "appointment",
        "instructions": "You book appointments for Sunrise Clinic. Context: [[CALLER_CONTEXT]]",
        "greeting": "Hello from Sunrise Clinic.",
        "knowledge": "",
        "version": 1,
        **overrides,
    }


def test_test_call_creates_the_agent_from_a_persona_and_places_the_call(client):
    client.app.state.services.settings.enable_outbound_calls = True
    body = {"clientId": "client-1", "persona": _persona(), "to": "+919876543210", "contactName": "Asha"}

    placed = client.post("/calls/test", json=body, headers=ADMIN)

    assert placed.status_code == 201, placed.text
    created = placed.json()
    assert created["status"] == "initiated" and created["contact_name"] == "Asha"
    agents = client.get("/agents", headers=ADMIN).json()
    agent = next(a for a in agents if a["persona_key"] == "clinic-test" and a["client_id"] == "client-1")
    assert agent["id"] == created["agent_id"]
    assert agent["name"] == "Clinic" and agent["workflow"] == "appointment"


def test_test_call_reuses_the_same_agent_on_a_second_call(client):
    client.app.state.services.settings.enable_outbound_calls = True
    body = {"clientId": "client-1", "persona": _persona(), "to": "+919876543210"}
    first = client.post("/calls/test", json=body, headers=ADMIN).json()
    client.post("/exotel/status?token=test-hook", data={"CallSid": first["exotel_call_sid"], "Status": "no-answer"})

    body["to"] = "+919876543211"
    second = client.post("/calls/test", json=body, headers=ADMIN).json()

    assert first["agent_id"] == second["agent_id"]
    agents = [a for a in client.get("/agents", headers=ADMIN).json() if a["persona_key"] == "clinic-test"]
    assert len(agents) == 1  # updated in place, not duplicated


def test_test_call_is_rejected_for_an_unspeakable_persona(client):
    client.app.state.services.settings.enable_outbound_calls = True
    del client.app.state.services.settings.bakbak_voices["te"]  # simulate BAKBAK_VOICES missing this language
    body = {"clientId": "client-1", "persona": _persona(language="te"), "to": "+919876543210"}

    rejected = client.post("/calls/test", json=body, headers=ADMIN)

    assert rejected.status_code == 422


def test_test_call_respects_outbound_disabled_and_capacity(client):
    body = {"clientId": "client-1", "persona": _persona(), "to": "+919876543210"}
    assert client.post("/calls/test", json=body, headers=ADMIN).status_code == 409  # outbound disabled

    client.app.state.services.settings.enable_outbound_calls = True
    assert client.post("/calls/test", json=body, headers=ADMIN).status_code == 201
    assert client.post("/calls/test", json=body, headers=ADMIN).status_code == 409  # capacity held while ringing


def test_test_call_applies_the_personas_outcome_rules_to_the_agent(client):
    client.app.state.services.settings.enable_outbound_calls = True
    persona = _persona(
        key="outcome-persona",
        outcomeRules=[{"name": "Interested", "definition": "Clear positive response."}],
        outcomeFallback="No decision.",
    )
    body = {"clientId": "client-1", "persona": persona, "to": "+919876543210"}

    placed = client.post("/calls/test", json=body, headers=ADMIN)

    assert placed.status_code == 201, placed.text
    agent = client.get(f"/agents/{placed.json()['agent_id']}", headers=ADMIN).json()
    assert agent["outcome_rules"] == [{"name": "Interested", "definition": "Clear positive response."}]
    assert agent["outcome_fallback"] == "No decision."


def test_test_call_rejects_an_outcome_rule_named_inconclusive(client):
    client.app.state.services.settings.enable_outbound_calls = True
    persona = _persona(outcomeRules=[{"name": "Inconclusive", "definition": "x"}], outcomeFallback="y")
    body = {"clientId": "client-1", "persona": persona, "to": "+919876543210"}

    rejected = client.post("/calls/test", json=body, headers=ADMIN)

    assert rejected.status_code == 422


def test_test_call_applies_the_chosen_voice_to_the_agent(client):
    client.app.state.services.settings.enable_outbound_calls = True
    body = {
        "clientId": "client-1",
        "persona": _persona(key="voiced-persona"),
        "to": "+919876543210",
        "voiceId": "chosen-voice",
        "ttsModelId": "m1",
    }

    placed = client.post("/calls/test", json=body, headers=ADMIN)

    assert placed.status_code == 201, placed.text
    call = client.portal.call(
        client.app.state.services.pool.fetchrow,
        "SELECT voice_id, tts_model_id FROM calls WHERE id = $1",
        placed.json()["id"],
    )
    assert (call["voice_id"], call["tts_model_id"]) == ("chosen-voice", "m1")
    agent = client.get(f"/agents/{placed.json()['agent_id']}", headers=ADMIN).json()
    assert agent["voice_id"] is None  # the choice belongs to this call, not to the shared persona agent


def test_test_call_rejects_a_half_set_voice_choice(client):
    client.app.state.services.settings.enable_outbound_calls = True
    body = {"clientId": "client-1", "persona": _persona(), "to": "+919876543210", "voiceId": "only-this"}

    rejected = client.post("/calls/test", json=body, headers=ADMIN)

    assert rejected.status_code == 422 and "together" in rejected.json()["detail"]


def test_test_call_rejects_an_unsupported_model(client):
    client.app.state.services.settings.enable_outbound_calls = True
    body = {
        "clientId": "client-1",
        "persona": _persona(),
        "to": "+919876543210",
        "voiceId": "v",
        "ttsModelId": "not-a-real-model",
    }

    rejected = client.post("/calls/test", json=body, headers=ADMIN)

    assert rejected.status_code == 422 and "Unsupported Bakbak model" in rejected.json()["detail"]


def test_a_later_test_call_with_no_voice_choice_does_not_inherit_an_earlier_one(client):
    client.app.state.services.settings.enable_outbound_calls = True
    persona = _persona(key="sticky-voice-persona")
    first = client.post(
        "/calls/test",
        json={"clientId": "client-1", "persona": persona, "to": "+919876543210", "voiceId": "v1", "ttsModelId": "m1"},
        headers=ADMIN,
    )
    client.post(
        "/exotel/status?token=test-hook",
        data={"CallSid": first.json()["exotel_call_sid"], "Status": "no-answer"},
    )

    second = client.post(
        "/calls/test", json={"clientId": "client-1", "persona": persona, "to": "+919876543211"}, headers=ADMIN
    )

    assert second.status_code == 201, second.text
    call = client.portal.call(
        client.app.state.services.pool.fetchrow,
        "SELECT voice_id, tts_model_id FROM calls WHERE id = $1",
        second.json()["id"],
    )
    assert (call["voice_id"], call["tts_model_id"]) == (None, None)
    agent = client.get(f"/agents/{second.json()['agent_id']}", headers=ADMIN).json()
    assert agent["voice_id"] is None


def test_media_stream_rejects_missing_credentials(client):
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/exotel/stream") as ws:
        ws.receive_text()


def converse(client, call_sid: str, to_number: str | None = None, **start_fields) -> None:
    """Play Exotel: connect, hear the greeting, speak once, hear the reply and the agent hanging up."""
    media = {"event": "media", "stream_sid": "MZstream", "media": {"payload": base64.b64encode(bytes(3200)).decode()}}
    with client.websocket_connect("/exotel/stream", headers=EXOTEL_AUTH) as ws:
        ws.send_text(json.dumps({"event": "connected"}))
        ws.send_text(exotel_start(call_sid, **({"to_number": to_number} if to_number else {}), **start_fields))
        audio_chunks = marks = 0
        while marks < 2:  # greeting, then the answer
            message = ws.receive_json()
            if message["event"] == "media":
                audio_chunks += 1
            elif message["event"] == "mark":
                ws.send_json({"event": "mark", "stream_sid": "MZstream", "mark": message["mark"]})
                marks += 1
                if marks == 1:
                    ws.send_json(media)  # caller speaks after the greeting
        assert audio_chunks > 0
        # The agent ended the call: the server closes the socket.
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_outbound_call_carries_the_contact_name_into_the_prompt(client):
    agent = client.post("/agents", json=APPOINTMENT_AGENT, headers=ADMIN).json()
    client.app.state.services.settings.enable_outbound_calls = True
    body = {"agent_id": agent["id"], "to": "+919876543210", "contact_name": "  Ravi \n Kumar "}
    placed = client.post("/calls", json=body, headers=ADMIN)
    assert placed.status_code == 201 and placed.json()["contact_name"] == "Ravi Kumar"

    converse(client, "CAout1")

    system_prompt = client.app.state.services.llm.requests[0][0]["content"]
    assert "Context: Outbound call to Ravi Kumar." in system_prompt and "[[" not in system_prompt


def test_inbound_call_end_to_end_is_persisted(client):
    client.post("/agents", json=APPOINTMENT_AGENT, headers=ADMIN)
    converse(client, "CAin1")
    system_prompt = client.app.state.services.llm.requests[0][0]["content"]
    assert "Context: Inbound call; the caller's name is not known." in system_prompt

    deadline = time.monotonic() + 5
    while True:
        calls = client.get("/calls", headers=ADMIN).json()
        if calls and calls[0]["outcome"] or time.monotonic() > deadline:
            break
        time.sleep(0.05)
    call = client.get(f"/calls/{calls[0]['id']}", headers=ADMIN).json()
    assert call["direction"] == "inbound" and call["status"] == "completed"
    assert call["end_reason"] == "agent_ended" and call["from_number"] == "+919876543210"
    assert [t["user_text"] for t in call["turns"]] == [None, "I need an appointment tomorrow at ten"]
    assert call["turns"][1]["agent_text"] == "Sure, may I have your name?"
    assert call["outcome"] == "appointment_requested"
    assert call["outcome_data"]["preferred_time"] == "tomorrow 10am"
    assert call["usage"]["asr_seconds"] == 0.2 and call["usage"]["llm_completion_tokens"] == 70
    assert call["usage"]["tts_provider"] == "bakbak" and call["usage"]["tts_model_id"] == "m1"
    assert call["usage"]["fallback_tts_characters"] == 0
    chars = call["usage"]["tts_characters"]  # greeting + reply, each priced at INR 2.5 per 1k characters
    assert call["cost"]["tts"] == round(chars / 1000 * 2.5, 4) and call["cost"]["complete"] is True

    # A second stream for the same CallSid is refused rather than merged into the finished call.
    with client.websocket_connect("/exotel/stream", headers=EXOTEL_AUTH) as ws:
        ws.send_text(exotel_start("CAin1"))
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()
    assert len(client.get("/calls", headers=ADMIN).json()) == 1
    assert call["latency"]["turns"] == 2 and call["latency"]["response_ms_avg"] is not None
