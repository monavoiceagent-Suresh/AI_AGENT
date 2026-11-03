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
