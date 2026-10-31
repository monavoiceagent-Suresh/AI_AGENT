"""Deepgram Aura TTS adapter: protocol mapping against a local fake of the real WebSocket API, and the
hard language restriction (Aura has no Hindi/Tamil/Telugu voices)."""

import json
from urllib.parse import parse_qs, urlsplit

import pytest
from websockets.asyncio.server import serve

from app.interfaces import TTSError
from app.providers.deepgram_tts import DeepgramTTS

pytestmark = pytest.mark.anyio

KEY = "deepgram-test-key"


class FakeDeepgramSpeak:
    def __init__(self, audio_chunks: list[bytes], *, warning: str | None = None):
        self.audio_chunks = audio_chunks
        self.warning = warning
        self.received_text: list[str] = []
        self.received_types: list[str] = []
        self.auth_header: str | None = None
        self.query: dict[str, list[str]] = {}
        self.closed_cleanly = False

    async def handler(self, ws):
        self.auth_header = ws.request.headers.get("Authorization")
        self.query = parse_qs(urlsplit(ws.request.path).query)
        async for raw in ws:
            message = json.loads(raw)
            self.received_types.append(message["type"])
            if message["type"] == "Speak":
                self.received_text.append(message["text"])
            elif message["type"] == "Flush":
                for chunk in self.audio_chunks:
                    await ws.send(chunk)
                if self.warning:
                    await ws.send(json.dumps({"type": "Warning", "description": self.warning}))
                else:
                    await ws.send(json.dumps({"type": "Flushed", "sequence_id": 0}))
            elif message["type"] == "Close":
                self.closed_cleanly = True
                return


async def synthesize(fake: FakeDeepgramSpeak, *, language="en", voice_id="aura-2-thalia-en") -> bytes:
    async with serve(fake.handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        tts = DeepgramTTS(api_key=KEY, url=f"ws://127.0.0.1:{port}/v1/speak")
        audio = b"".join(
            [
                chunk
                async for chunk in tts.stream(
                    "Hello there",
                    voice_id=voice_id,
                    model_id="aura",
                    language=language,
                    sample_rate=16000,
                    previous_text="",
                )
            ]
        )
    return audio


async def test_sends_speak_then_flush_and_yields_raw_pcm_unmodified():
    fake = FakeDeepgramSpeak([b"\x01\x00\x02\x00", b"\x03\x00\x04\x00"])

    audio = await synthesize(fake)

    assert audio == b"\x01\x00\x02\x00\x03\x00\x04\x00"  # passed through byte-for-byte, no float32 decoding
    assert fake.received_text == ["Hello there"]
    assert fake.received_types == ["Speak", "Flush", "Close"]
    assert fake.closed_cleanly
    assert fake.auth_header == f"Token {KEY}"
    assert fake.query["model"] == ["aura-2-thalia-en"]
    assert fake.query["encoding"] == ["linear16"]
    assert fake.query["sample_rate"] == ["16000"]


async def test_a_warning_message_raises_tts_error():
    fake = FakeDeepgramSpeak([b"\x01\x00"], warning="something went wrong")

    with pytest.raises(TTSError, match="something went wrong"):
        await synthesize(fake)


async def test_no_audio_at_all_raises_tts_error():
    fake = FakeDeepgramSpeak([])

    with pytest.raises(TTSError, match="no audio"):
        await synthesize(fake)


@pytest.mark.parametrize("language", ["hi", "ta", "te"])
async def test_non_english_languages_are_refused_without_even_connecting(language):
    fake = FakeDeepgramSpeak([b"\x01\x00"])

    with pytest.raises(TTSError, match="cannot speak language"):
        await synthesize(fake, language=language)

    assert fake.received_types == []  # never connected: no Speak/Flush/Close reached the (fake) server
