import base64
import json
import struct

import httpx
import pytest

from app.calls import fallback_voice_for, validate_agent, voice_for
from app.config import Settings
from app.interfaces import TTSError, Voice
from app.models import AgentIn
from app.providers.bakbak_tts import BakbakTTS, float32_to_pcm16
from tests.fakes import make_settings

pytestmark = pytest.mark.anyio


def sse(*events: dict) -> bytes:
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)


def f32(*samples: float) -> bytes:
    return struct.pack(f"<{len(samples)}f", *samples)


def chunk(raw: bytes) -> dict:
    return {"type": "chunk", "status_code": 206, "done": False, "data": base64.b64encode(raw).decode()}


DONE = {"type": "done", "status_code": 200, "done": True}


def test_float32_to_pcm16_scales_and_clips():
    pcm, rest = float32_to_pcm16(f32(0.0, 0.5, -0.5, 1.0, 1.5, -2.0))
    assert struct.unpack("<6h", pcm) == (0, 16384, -16384, 32767, 32767, -32768)
    assert rest == b""


def test_float32_to_pcm16_carries_a_split_sample_to_the_next_chunk():
    raw = f32(0.25, -0.25)
    first, carry = float32_to_pcm16(raw[:6])  # 1.5 samples
    second, carry = float32_to_pcm16(carry + raw[6:])
    assert struct.unpack("<h", first) == (8192,) and struct.unpack("<h", second) == (-8192,) and carry == b""


async def synthesize(handler, **overrides) -> tuple[bytes, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(record)) as http:
        tts = BakbakTTS(http, api_key="secret-key", base_url="https://hub.example/v1/")
        args = dict(voice_id="v1", model_id="standard", language="ta", sample_rate=8000, previous_text="") | overrides
        return b"".join([c async for c in tts.stream("வணக்கம்", **args)]), seen


async def test_stream_sends_documented_request_and_decodes_sse_audio():
    raw = f32(0.5, -0.5, 0.25, 0.0)

    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            # A float split across two chunk events must be reassembled.
            content=sse(chunk(raw[:6]), chunk(raw[6:]), DONE),
        )

    pcm, seen = await synthesize(handler)

    assert struct.unpack("<4h", pcm) == (16384, -16384, 8192, 0)
    request = seen[0]
    assert str(request.url) == "https://hub.example/v1/text-to-speech/stream"
    assert request.headers["x-api-key"] == "secret-key"
    assert json.loads(request.content) == {
        "text": "வணக்கம்",
        "voice_id": "v1",
        "model": "standard",
        "language": "ta",
        "codec": "pcm",
        "sample_rate": 8000,
    }


async def test_english_agents_use_the_indian_english_language_code():
    _, seen = await synthesize(lambda r: httpx.Response(200, content=sse(DONE)), language="en", model_id="m1")
    assert json.loads(seen[0].content)["language"] == "en-in" and json.loads(seen[0].content)["model"] == "m1"


async def test_http_error_raises_tts_error_without_leaking_the_key():
    def handler(request):
        return httpx.Response(404, json={"detail": "unknown voice_id: 'v1'"})

    with pytest.raises(TTSError, match="404") as exc:
        await synthesize(handler)
    assert "secret-key" not in str(exc.value)


async def test_error_event_and_truncated_stream_raise():
    error = {"type": "error", "message": "model crashed"}
    with pytest.raises(TTSError, match="stream error"):
        await synthesize(lambda r: httpx.Response(200, content=sse(chunk(f32(0.1)), error)))
    with pytest.raises(TTSError, match="without a done"):
        await synthesize(lambda r: httpx.Response(200, content=sse(chunk(f32(0.1)))))


def agent_in(**overrides) -> AgentIn:
    base = dict(name="A", language="ta", workflow="general", instructions="i", greeting="g")
    return AgentIn(**(base | overrides))


def test_voice_defaults_come_from_the_language_map_and_agents_can_override():
    cfg = make_settings(tts_provider="bakbak")
    assert voice_for(agent_in(language="ta"), cfg) == Voice(voice_id="ta-voice", model_id="standard")
    assert voice_for(agent_in(language="en"), cfg) == Voice(voice_id="en-voice", model_id="m1")
    custom = agent_in(voice_id="custom", tts_model_id="m1")
    assert voice_for(custom, cfg) == Voice(voice_id="custom", model_id="m1")


def test_bakbak_agent_validation():
    cfg = make_settings(tts_provider="bakbak", bakbak_voices={"ta": {"voice_id": "t", "model_id": "standard"}})
    assert validate_agent(agent_in(language="ta"), cfg) is None
    assert "no voice for language 'hi'" in validate_agent(agent_in(language="hi"), cfg)
    assert "together" in validate_agent(agent_in(voice_id="x"), cfg)
    assert "together" in validate_agent(agent_in(tts_model_id="m1"), cfg)
    assert "Unsupported Bakbak model" in validate_agent(agent_in(voice_id="x", tts_model_id="eleven_v3"), cfg)
    assert validate_agent(agent_in(voice_id="x", tts_model_id="m1"), cfg) is None


def test_elevenlabs_provider_keeps_its_language_rules():
    cfg = make_settings(tts_provider="elevenlabs_standard")
    assert "cannot speak language 'te'" in validate_agent(agent_in(language="te"), cfg)
    assert voice_for(agent_in(language="hi"), cfg) == Voice(voice_id="el-voice", model_id="eleven_flash_v2_5")


def test_fallback_voice_only_when_configured_and_the_language_is_supported():
    assert fallback_voice_for(agent_in(language="ta"), make_settings(tts_fallback_provider=None)) is None
    cfg = make_settings(tts_provider="bakbak", tts_fallback_provider="elevenlabs_standard")
    assert fallback_voice_for(agent_in(language="ta"), cfg) == Voice(voice_id="el-voice", model_id="eleven_flash_v2_5")
    assert fallback_voice_for(agent_in(language="te"), cfg) is None  # Flash v2.5 cannot speak Telugu


def test_settings_reject_an_unusable_primary_tts_configuration():
    with pytest.raises(ValueError, match="missing its required credentials"):
        make_settings(tts_provider="bakbak", bakbak_api_key=None)
    with pytest.raises(ValueError, match="missing its required credentials"):
        make_settings(tts_provider="bakbak", bakbak_voices={})
    with pytest.raises(ValueError, match="missing its required credentials"):
        make_settings(elevenlabs_tts_voice_id="")
    assert isinstance(
        make_settings(tts_provider="elevenlabs_standard", bakbak_api_key=None, bakbak_voices={}), Settings
    )
