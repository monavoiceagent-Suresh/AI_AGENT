"""In-memory provider fakes for exercising the orchestrator without network access."""

import asyncio
import json
from collections.abc import AsyncIterator

from app.config import Settings
from app.interfaces import (
    CallEnded,
    CallEvent,
    CommittedTranscript,
    LLMUsage,
    PartialTranscript,
    PlaybackMark,
)


class FakeMedia:
    """Telephony leg that acknowledges every mark immediately, like Exotel after playback."""

    sample_rate = 8000

    def __init__(self) -> None:
        self.inbox: asyncio.Queue[CallEvent] = asyncio.Queue()
        self.audio = bytearray()
        self.marks: list[str] = []
        self.clears = 0
        self.ack_marks = True

    async def events(self) -> AsyncIterator[CallEvent]:
        while True:
            event = await self.inbox.get()
            yield event
            if isinstance(event, CallEnded):
                return

    async def send_audio(self, pcm: bytes) -> None:
        self.audio.extend(pcm)

    async def flush_audio(self) -> None:
        pass

    async def mark(self, name: str) -> None:
        self.marks.append(name)
        if self.ack_marks:
            self.inbox.put_nowait(PlaybackMark(name))

    async def clear(self) -> None:
        self.clears += 1

    def playback_remaining(self) -> float:
        return 0.0

    def hang_up(self) -> None:
        self.inbox.put_nowait(CallEnded("callended"))


class FakeASR:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[PartialTranscript | CommittedTranscript] = asyncio.Queue()
        self.audio_bytes = 0

    async def send_audio(self, pcm: bytes) -> None:
        self.audio_bytes += len(pcm)

    async def events(self) -> AsyncIterator[PartialTranscript | CommittedTranscript]:
        while True:
            yield await self.queue.get()

    def say(self, text: str) -> None:
        self.queue.put_nowait(PartialTranscript(text))
        self.queue.put_nowait(CommittedTranscript(text))


class FakeLLM:
    def __init__(self, replies: list[str], token_delay: float = 0.0, outcome: str = "{}") -> None:
        self.replies = list(replies)
        self.token_delay = token_delay
        self.outcome = outcome
        self.requests: list[list[dict[str, str]]] = []

    async def stream(self, messages: list[dict[str, str]], model: str) -> AsyncIterator[str | LLMUsage]:
        self.requests.append(messages)
        reply = self.replies.pop(0)
        for word in reply.split(" "):
            await asyncio.sleep(self.token_delay)
            yield word + " "
        yield LLMUsage(prompt_tokens=100, completion_tokens=20, cost_usd=0.0001)

    async def complete(self, messages: list[dict[str, str]], model: str) -> tuple[str, LLMUsage]:
        return self.outcome, LLMUsage(prompt_tokens=200, completion_tokens=50, cost_usd=0.0002)


class FakeTTS:
    """Returns 100 ms of silence per word, optionally slowly. Records the text and voice of each request."""

    def __init__(self, chunk_delay: float = 0.0) -> None:
        self.chunk_delay = chunk_delay
        self.texts: list[str] = []
        self.voices: list[tuple[str, str]] = []

    async def stream(
        self, text: str, *, sample_rate: int, voice_id: str = "", model_id: str = "", **_: object
    ) -> AsyncIterator[bytes]:
        self.texts.append(text)
        self.voices.append((voice_id, model_id))
        for _word in text.split():
            await asyncio.sleep(self.chunk_delay)
            yield bytes(sample_rate * 2 // 10)


def make_settings(**overrides) -> Settings:
    base = dict(
        app_secret_key="s",
        public_base_url="https://example.test",
        exotel_account_sid="sid",
        exotel_api_key="k",
        exotel_api_token="t",
        exotel_caller_id="080",
        exotel_webhook_secret="w",
        elevenlabs_api_key="e",
        elevenlabs_tts_voice_id="el-voice",
        openrouter_api_key="o",
        llm_provider="openrouter",
        openrouter_fallback_model="fallback-model",
        database_url="postgresql://x",
        bakbak_api_key="b",
        deepgram_api_key="dg",
        bakbak_voices={
            "en": {"voice_id": "en-voice", "model_id": "m1"},
            "hi": {"voice_id": "hi-voice", "model_id": "m1"},
            "ta": {"voice_id": "ta-voice", "model_id": "standard"},
            "te": {"voice_id": "te-voice", "model_id": "standard"},
        },
        usd_to_inr=85.0,
        exotel_inr_per_minute=None,
        # No TTS/ASR fallback by default: tests that exercise a fallback set tts_fallback_provider /
        # asr_fallback_provider explicitly, so a test that only picks a primary provider never risks an
        # accidental TTS_PROVIDER == TTS_FALLBACK_PROVIDER collision.
        tts_fallback_provider=None,
        asr_fallback_provider=None,
    )
    return Settings(_env_file=None, **(base | overrides))


def exotel_start(
    call_sid: str,
    sample_rate: int = 8000,
    to_number: str = "+918047491899",
    from_number: str = "+919876543210",
    extra: dict | None = None,
    custom_parameters: dict | None = None,
) -> str:
    return json.dumps(
        {
            "event": "start",
            "sequence_number": "1",
            "stream_sid": "MZstream",
            "start": {
                "stream_sid": "MZstream",
                "call_sid": call_sid,
                "account_sid": "ACtest",
                "from": from_number,
                "to": to_number,
                "custom_parameters": custom_parameters or {},
                "media_format": {"encoding": "audio/x-raw", "sample_rate": str(sample_rate), "bit_rate": "16"},
                **(extra or {}),
            },
        }
    )
