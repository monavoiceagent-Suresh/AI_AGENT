"""GET /voices proxies the configured TTS provider's real voice list; it never hardcodes one."""

import httpx
import pytest

from app.voices import VoicesUnavailable, list_voices
from tests.fakes import make_settings

pytestmark = pytest.mark.anyio


def client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_bakbak_voices_are_fetched_with_the_api_key_and_parsed():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers["x-api-key"]))
        return httpx.Response(
            200,
            json=[
                {"voice_id": "v1", "name": "Anitha", "language": "ta", "model_id": "standard"},
                {"voice_id": "v2", "name": "Rahul", "language": "en", "model_id": "m1"},
                {"id": "not-malformed-just-missing-voice_id-key"},  # skipped: no usable voice_id
            ],
        )

    settings = make_settings(tts_provider="bakbak", bakbak_base_url="https://bakbak.test/v1")
    result = await list_voices(client(handler), settings)

    assert seen == [("https://bakbak.test/v1/voices", "b")]
    assert result.provider == "bakbak"
    assert [v.voice_id for v in result.voices] == ["v1", "v2"]
    assert result.voices[0].name == "Anitha" and result.voices[0].model_id == "standard"


async def test_bakbak_voices_accepts_a_wrapped_list_too():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"voices": [{"voice_id": "v1", "name": "Anitha"}]})

    settings = make_settings(tts_provider="bakbak")
    result = await list_voices(client(handler), settings)

    assert [v.voice_id for v in result.voices] == ["v1"]
    assert result.voices[0].language is None and result.voices[0].model_id is None


async def test_elevenlabs_voices_use_the_xi_api_key_header_and_the_label_language():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers["xi-api-key"]))
        return httpx.Response(
            200,
            json={"voices": [{"voice_id": "e1", "name": "Rachel", "labels": {"language": "en", "age": "young"}}]},
        )

    settings = make_settings(tts_provider="elevenlabs_standard")
    result = await list_voices(client(handler), settings)

    assert seen == [("https://api.elevenlabs.io/v1/voices", "e")]
    assert result.provider == "elevenlabs"
    assert result.voices[0].voice_id == "e1"
    assert result.voices[0].name == "Rachel"
    assert result.voices[0].language == "en"
    assert result.voices[0].model_id is None  # ElevenLabs voices are not tied to one model


async def test_elevenlabs_premium_tier_uses_the_same_vendor_voice_list():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"voices": [{"voice_id": "e1", "name": "Rachel"}]})

    result = await list_voices(client(handler), make_settings(tts_provider="elevenlabs_premium"))

    assert result.provider == "elevenlabs" and result.voices[0].voice_id == "e1"


async def test_deepgram_aura_returns_its_fixed_english_voice_catalogue_with_no_http_call():
    def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("Deepgram Aura has no list-voices endpoint; this should never be called")

    result = await list_voices(client(handler), make_settings(tts_provider="deepgram_aura"))

    assert result.provider == "deepgram"
    assert all(v.voice_id.startswith("aura-2-") and v.voice_id.endswith("-en") for v in result.voices)
    assert all(v.language == "en" for v in result.voices)
    assert len(result.voices) > 30


async def test_a_connection_failure_becomes_voicesunavailable_not_a_crash():
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(VoicesUnavailable):
        await list_voices(client(handler), make_settings(tts_provider="bakbak"))


async def test_an_error_response_becomes_voicesunavailable():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "bad key"})

    with pytest.raises(VoicesUnavailable):
        await list_voices(client(handler), make_settings(tts_provider="bakbak"))


async def test_an_unreadable_body_becomes_voicesunavailable():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    with pytest.raises(VoicesUnavailable):
        await list_voices(client(handler), make_settings(tts_provider="bakbak"))


async def test_a_malformed_entry_with_no_name_is_skipped_not_fabricated():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"voice_id": "v1"}, {"voice_id": "v2", "name": "Rahul"}])

    settings = make_settings(tts_provider="bakbak")
    result = await list_voices(client(handler), settings)

    assert [v.voice_id for v in result.voices] == ["v2"]
