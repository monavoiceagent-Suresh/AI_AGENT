import io
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app import api, voices
from app.models import VoiceOption, VoicesOut
from app.providers.exotel import ExotelClient
from tests.fakes import make_settings


@pytest.mark.anyio
async def test_recording_redirect_does_not_forward_exotel_credentials():
    seen = []

    def serve(request):
        seen.append(request)
        if request.url.host == "api.exotel.com":
            return httpx.Response(302, headers={"Location": "https://s3.amazonaws.com/exotelrecordings/sample.mp3"})
        return httpx.Response(200, headers={"Content-Type": "audio/mpeg"}, content=b"sample")

    async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as http:
        exotel = ExotelClient(http, base_url="https://api.exotel.com", account_sid="test", api_key="test",
                              api_token="test", caller_id="unused", app_id="unused")
        assert await exotel.open_recording("https://api.exotel.com.evil.example/sample") is None
        audio = await exotel.open_recording("https://api.exotel.com/recordings/sample")
        assert audio is not None
        assert await audio.aread() == b"sample"
        await audio.aclose()
    assert len(seen) == 2
    assert "authorization" in seen[0].headers
    assert "authorization" not in seen[1].headers


@pytest.mark.anyio
async def test_recording_url_given_as_plain_http_is_fetched_over_https():
    seen = []

    def serve(request):
        seen.append(request)
        assert request.url.scheme == "https"
        return httpx.Response(200, headers={"Content-Type": "audio/mpeg"}, content=b"sample")

    async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as http:
        exotel = ExotelClient(http, base_url="https://api.exotel.com", account_sid="test", api_key="test",
                              api_token="test", caller_id="unused", app_id="unused")
        audio = await exotel.open_recording("http://api.exotel.com/recordings/sample")
        assert audio is not None
        assert await audio.aread() == b"sample"
        await audio.aclose()
    assert len(seen) == 1


@pytest.mark.anyio
async def test_recording_checks_tenant_before_fetching_and_refreshes_expired_url():
    old_url = "https://s3.amazonaws.com/exotelrecordings/old.mp3"
    signed_url = "https://s3.amazonaws.com/exotelrecordings/new.mp3"
    number = "+918000000001"
    row = {"client_id": "workspace", "direction": "inbound", "exotel_call_sid": "CA-test",
           "recording_url": old_url, "dialled_number": number, "to_number": None}
    exotel = SimpleNamespace(
        open_recording=AsyncMock(side_effect=[None, httpx.Response(200, content=b"audio")]),
        recording_urls=AsyncMock(return_value=(old_url, signed_url)),
    )
    svc = SimpleNamespace(pool=SimpleNamespace(fetchrow=AsyncMock(return_value=row)), exotel=exotel)
    with pytest.raises(api.HTTPException) as denied:
        await api.call_recording(1, svc, "other-workspace", "inbound", number)
    assert denied.value.status_code == 404
    exotel.open_recording.assert_not_awaited()

    response = await api.call_recording(1, svc, "workspace", "inbound", number)
    assert b"".join([part async for part in response.body_iterator]) == b"audio"
    assert response.headers["cache-control"] == "private, no-store"
    assert [call.args[0] for call in exotel.open_recording.await_args_list] == [old_url, signed_url]


@pytest.mark.anyio
@pytest.mark.parametrize(("provider", "vendor", "model", "language"), [
    ("elevenlabs_premium", "elevenlabs", None, "ta"),
    ("elevenlabs_standard", "elevenlabs", None, "en"),
    ("deepgram_aura", "deepgram", "aura-2", "en"),
    ("bakbak", "bakbak", "m1", "hi"),
])
async def test_preview_uses_catalogue_voice_and_returns_pcm_wav(monkeypatch, provider, vendor, model, language):
    settings = make_settings(tts_provider=provider, elevenlabs_tts_voice_id="default")
    catalogue = VoicesOut(provider=vendor, voices=[
        VoiceOption(voice_id="selected", name="Sample", model_id=model, language=language),
    ])
    monkeypatch.setattr(voices, "list_voices", AsyncMock(return_value=catalogue))
    selected = []

    async def stream(_text, **kwargs):
        selected.append(kwargs)
        yield b"\x00\x01"

    svc = SimpleNamespace(http=None, settings=settings, tts=SimpleNamespace(stream=stream))
    assert await voices.preview_voice(svc, "unknown") is None
    wav = await voices.preview_voice(svc, "selected")
    assert selected[0]["voice_id"] == "selected"
    assert selected[0]["language"] == language
    assert selected[0]["model_id"] == (model or (
        settings.elevenlabs_premium_model_id if provider == "elevenlabs_premium" else settings.elevenlabs_tts_model_id
    ))
    with wave.open(io.BytesIO(wav), "rb") as audio:
        assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 16000)
        assert audio.readframes(1) == b"\x00\x01"
