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


class FakeAllocator:
    """A minimal in-memory stand-in for app.allocator.ChannelAllocator, for tests that exercise
    call-placement logic with no real Redis/Postgres. Mirrors just the behaviour callers depend
    on: an admission limit with the inbound-reserved carve-out, and idempotent reserve/release.
    """

    def __init__(self, *, max_concurrent_calls: int = 10, inbound_reserved_calls: int = 0) -> None:
        self.max_concurrent_calls = max_concurrent_calls
        self.inbound_reserved_calls = inbound_reserved_calls
        self.allocations: dict[str, str] = {}  # allocation id -> direction
        self.call_sids: dict[str, str] = {}  # call_sid -> allocation id

    async def reserve(
        self,
        allocation_id: str,
        direction: str,
        *,
        call_sid: str | None = None,
        client_id: str | None = None,
        route_phone: str | None = None,
        dispatch_lead_id: int | None = None,
    ) -> bool:
        if allocation_id in self.allocations:
            return True
        limit = self.max_concurrent_calls - (self.inbound_reserved_calls if direction == "outbound" else 0)
        if len(self.allocations) >= limit:
            return False
        self.allocations[allocation_id] = direction
        if call_sid:
            self.call_sids[call_sid] = allocation_id
        return True

    async def set_call_sid(self, allocation_id: str, call_sid: str) -> None:
        self.call_sids[call_sid] = allocation_id

    async def mark_connected(self, allocation_id: str) -> None:
        pass

    async def mark_connected_by_call_sid(self, call_sid: str) -> None:
        pass

    async def release(self, allocation_id: str) -> None:
        self.allocations.pop(allocation_id, None)

    async def release_by_call_sid(self, call_sid: str) -> None:
        allocation_id = self.call_sids.pop(call_sid, None)
        if allocation_id:
            self.allocations.pop(allocation_id, None)

    async def occupied(self) -> int:
        return len(self.allocations)


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
        redis_url="redis://x",
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
