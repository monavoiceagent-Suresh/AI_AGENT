from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app import api


@pytest.fixture
def callback_client():
    app = FastAPI()
    app.include_router(api.public)
    execute = AsyncMock(return_value="UPDATE 1")
    app.state.services = SimpleNamespace(
        settings=SimpleNamespace(exotel_webhook_secret=SecretStr("test-hook")),
        pool=SimpleNamespace(execute=execute),
        allocator=SimpleNamespace(release_by_call_sid=AsyncMock()),
    )
    with TestClient(app) as client:
        yield client, execute


@pytest.mark.parametrize("encoding", ["multipart", "data", "json"])
def test_callback_accepts_required_fields_and_maps_no_answer(callback_client, encoding):
    client, execute = callback_client
    fields = {"CallSid": "CA-test", "Status": "no-answer"}
    body = {"files": {key: (None, value) for key, value in fields.items()}} if encoding == "multipart" else {
        encoding: fields
    }
    assert client.post("/exotel/status", params={"token": "test-hook"}, **body).json() == {"ok": True}
    query, call_sid, status = execute.await_args.args
    assert "status = 'initiated'" in query
    assert (call_sid, status) == ("CA-test", "no_answer")


def test_callback_auth_and_missing_required_field(callback_client):
    client, execute = callback_client
    files = {"CallSid": (None, "CA-test")}
    assert client.post("/exotel/status", files=files).status_code == 403
    response = client.post("/exotel/status", params={"token": "test-hook"}, files=files)
    assert response.status_code == 422
    assert response.json() == {"detail": "CallSid and Status are required"}
    execute.assert_not_awaited()


def test_callback_acknowledges_duplicate_without_overwriting_call(callback_client):
    client, execute = callback_client
    execute.side_effect = ["UPDATE 1", "UPDATE 0"]
    for status in ("busy", "completed"):
        response = client.post(
            "/exotel/status", params={"token": "test-hook"},
            files={"CallSid": (None, "CA-test"), "Status": (None, status)},
        )
        assert response.status_code == 200
    assert all("status = 'initiated'" in call.args[0] for call in execute.await_args_list)


def test_terminal_callback_stores_recording_even_after_media_completion(callback_client):
    client, execute = callback_client
    execute.side_effect = ["UPDATE 0", "UPDATE 1"]
    recording_url = "https://s3.amazonaws.com/exotelrecordings/test.mp3"
    response = client.post("/exotel/status", params={"token": "test-hook"}, files={
        "CallSid": (None, "CA-test"), "Status": (None, "completed"), "RecordingUrl": (None, recording_url),
    })
    assert response.status_code == 200
    query, sid, url = execute.await_args.args
    assert "SET recording_url" in query
    assert (sid, url) == ("CA-test", recording_url)
