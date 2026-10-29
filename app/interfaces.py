"""Contracts between the voice-session orchestrator and the provider adapters."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict


@dataclass(frozen=True)
class CallAudio:
    pcm: bytes  # 16-bit little-endian mono PCM at CallMedia.sample_rate


@dataclass(frozen=True)
class PlaybackMark:
    name: str


@dataclass(frozen=True)
class CallEnded:
    reason: str


CallEvent = CallAudio | PlaybackMark | CallEnded


class CallMedia(Protocol):
    sample_rate: int

    def events(self) -> AsyncIterator[CallEvent]: ...

    async def send_audio(self, pcm: bytes) -> None: ...

    async def flush_audio(self) -> None:
        """Send any partially buffered audio."""

    async def mark(self, name: str) -> None:
        """Ask for a PlaybackMark once everything sent so far has played."""

    async def clear(self) -> None:
        """Discard audio queued but not yet played."""

    def playback_remaining(self) -> float:
        """Seconds of sent audio expected to still be playing."""


@dataclass(frozen=True)
class PartialTranscript:
    text: str


@dataclass(frozen=True)
class CommittedTranscript:
    text: str


class Transcriber(Protocol):
    async def send_audio(self, pcm: bytes) -> None: ...

    def events(self) -> AsyncIterator[PartialTranscript | CommittedTranscript]: ...


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        return LLMUsage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.cost_usd + other.cost_usd,
        )


class LLM(Protocol):
    def stream(self, messages: list[dict[str, str]], model: str) -> AsyncIterator[str | LLMUsage]:
        """Yield text deltas, then a final LLMUsage."""

    async def complete(self, messages: list[dict[str, str]], model: str) -> tuple[str, LLMUsage]: ...


class TTSError(Exception):
    pass


class Voice(BaseModel):
    """A provider voice together with the synthesis model it belongs to."""

    model_config = ConfigDict(frozen=True)

    voice_id: str
    model_id: str


class TTS(Protocol):
    def stream(
        self,
        text: str,
        *,
        voice_id: str,
        model_id: str,
        language: str,
        sample_rate: int,
        previous_text: str,
    ) -> AsyncIterator[bytes]:
        """Yield 16-bit PCM audio at sample_rate."""
