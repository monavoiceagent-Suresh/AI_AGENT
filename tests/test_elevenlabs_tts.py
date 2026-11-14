import json

import httpx
import pytest

from app.providers.elevenlabs_tts import ElevenLabsTTS


@pytest.mark.anyio
@pytest.mark.parametrize("model_id", ["eleven_v3", "eleven_flash_v2_5"])
async def test_stream_sends_context_only_for_supported_models(model_id):
    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "next_text" not in body
        if model_id == "eleven_v3":
            assert "previous_text" not in body
        else:
            assert body["previous_text"] == "Earlier sentence."
        assert request.url.path.endswith("/stream")
        assert request.url.params["output_format"] == "pcm_16000"
        return httpx.Response(200, content=b"\x00\x01")

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        tts = ElevenLabsTTS(http, api_key="test", stability=0.5, similarity_boost=0.8)
        chunks = [
            chunk
            async for chunk in tts.stream(
                "Next sentence.",
                voice_id="voice",
                model_id=model_id,
                language="en",
                sample_rate=16000,
                previous_text="Earlier sentence.",
            )
        ]
    assert b"".join(chunks) == b"\x00\x01"


@pytest.mark.anyio
async def test_stream_voice_settings_override_the_constructors_defaults_per_field():
    seen = {}

    def respond(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content)["voice_settings"])
        return httpx.Response(200, content=b"\x00\x01")

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        tts = ElevenLabsTTS(http, api_key="test", stability=0.5, similarity_boost=0.8)
        async for _ in tts.stream(
            "Hello.",
            voice_id="voice",
            model_id="eleven_flash_v2_5",
            language="en",
            sample_rate=16000,
            previous_text="",
            voice_settings={"stability": 0.9, "style": 0.3},
        ):
            pass

    # stability/style came from the per-call override; similarity_boost kept the constructor default.
    assert seen == {"stability": 0.9, "similarity_boost": 0.8, "style": 0.3}


@pytest.mark.anyio
async def test_stream_with_no_voice_settings_override_keeps_the_constructors_defaults():
    seen = {}

    def respond(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content)["voice_settings"])
        return httpx.Response(200, content=b"\x00\x01")

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        tts = ElevenLabsTTS(http, api_key="test", stability=0.5, similarity_boost=0.8)
        async for _ in tts.stream(
            "Hello.", voice_id="voice", model_id="eleven_flash_v2_5", language="en", sample_rate=16000,
            previous_text="",
        ):
            pass

    assert seen == {"stability": 0.5, "similarity_boost": 0.8}
