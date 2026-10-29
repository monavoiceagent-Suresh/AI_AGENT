"""Voice-session orchestrator: caller audio -> ASR -> LLM -> TTS -> caller, with barge-in."""

import asyncio
import logging
import re
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from app.interfaces import (
    LLM,
    TTS,
    CallAudio,
    CallEnded,
    CallMedia,
    CommittedTranscript,
    LLMUsage,
    PlaybackMark,
    Transcriber,
    Voice,
)
from app.prompts import END_CALL_TOKEN

log = logging.getLogger(__name__)

SENTENCE_END = re.compile(r"[.!?।॥](?=\s)|\n")
CLAUSE_END = re.compile(r"[,;:](?=\s)")
# Start speaking at a clause boundary once this much text is waiting, rather than wait for a full stop.
SOFT_FLUSH_CHARS = 60
# Partial transcripts shorter than this (in letters/digits) don't interrupt the agent.
MIN_BARGE_IN_CHARS = 2
MARK_GRACE_SECONDS = 2.0


class SentenceSplitter:
    """Groups streamed LLM text into speakable sentences."""

    def __init__(self) -> None:
        self._buffer = ""

    def push(self, text: str) -> list[str]:
        self._buffer += text
        out: list[str] = []
        while True:
            match = SENTENCE_END.search(self._buffer)
            if match is None and len(self._buffer) >= SOFT_FLUSH_CHARS:
                match = next(reversed(list(CLAUSE_END.finditer(self._buffer))), None)
            if match is None:
                return out
            sentence, self._buffer = self._buffer[: match.end()].strip(), self._buffer[match.end() :]
            if sentence:
                out.append(sentence)

    def finish(self) -> list[str]:
        rest, self._buffer = self._buffer.strip(), ""
        return [rest] if rest else []


@dataclass(frozen=True)
class SessionConfig:
    system_prompt: str
    greeting: str
    fallback: str
    language: str
    voice: Voice
    fallback_voice: Voice | None
    llm_model: str
    barge_in: bool
    idle_timeout_seconds: float
    max_duration_seconds: float
    log_transcripts: bool


@dataclass
class _Response:
    id: int
    user_text: str | None
    committed_at: float
    endpointing_ms: int | None
    sent: list[str] = field(default_factory=list)  # sentences whose audio was sent
    played: int = 0  # sentences confirmed played via marks
    marked: asyncio.Event = field(default_factory=asyncio.Event)
    end_call: bool = False
    llm_ttft_ms: int | None = None
    tts_ttfb_ms: int | None = None
    response_latency_ms: int | None = None


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


def _join(*parts: str) -> str:
    return " ".join(p for p in parts if p).strip()


class VoiceSession:
    def __init__(
        self,
        config: SessionConfig,
        *,
        media: CallMedia,
        asr: Transcriber,
        llm: LLM,
        tts: TTS,
        fallback_tts: TTS | None = None,
    ):
        self.config = config
        self.media = media
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.fallback_tts = fallback_tts
        self.history: list[dict[str, str]] = [{"role": "system", "content": config.system_prompt}]
        self.turns: list[dict[str, Any]] = []
        self.asr_bytes = 0
        self.tts_characters = 0
        self.fallback_tts_characters = 0
        self.llm_usage = LLMUsage(0, 0, 0.0)
        self._pending_user = ""
        self._response: _Response | None = None
        self._response_task: asyncio.Task | None = None
        self._next_response_id = 0
        self._last_partial_at: float | None = None
        self._last_activity = time.monotonic()
        self._hangup = asyncio.Event()

    @property
    def asr_seconds(self) -> float:
        return self.asr_bytes / (2 * self.media.sample_rate)

    async def run(self) -> str:
        """Converse until the call ends; returns the end reason."""
        self._start_response(user_text=None, committed_at=time.monotonic(), endpointing_ms=None)
        tasks = [
            asyncio.create_task(self._pump_call()),
            asyncio.create_task(self._pump_asr()),
            asyncio.create_task(self._watchdog()),
            asyncio.create_task(self._wait_hangup()),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await self._cancel_response()
        finished = next(iter(done))
        if finished.exception() is not None:
            log.error("voice session failed", exc_info=finished.exception())
            return f"error:{type(finished.exception()).__name__}"
        return finished.result()

    # ---- inputs -------------------------------------------------------------------------

    async def _pump_call(self) -> str:
        async for event in self.media.events():
            if isinstance(event, CallAudio):
                self.asr_bytes += len(event.pcm)
                await self.asr.send_audio(event.pcm)
            elif isinstance(event, PlaybackMark):
                self._on_mark(event.name)
            elif isinstance(event, CallEnded):
                return f"caller:{event.reason}"
        return "caller:disconnected"

    async def _pump_asr(self) -> str:
        async for event in self.asr.events():
            if isinstance(event, CommittedTranscript):
                await self._on_committed(event.text)
            else:
                await self._on_partial(event.text)
        return "asr_closed"

    async def _watchdog(self) -> str:
        started = time.monotonic()
        while True:
            await asyncio.sleep(1)
            now = time.monotonic()
            if now - started >= self.config.max_duration_seconds:
                return "max_duration"
            if not self._responding() and now - self._last_activity >= self.config.idle_timeout_seconds:
                return "idle_timeout"

    async def _wait_hangup(self) -> str:
        await self._hangup.wait()
        return "agent_ended"

    def _on_mark(self, name: str) -> None:
        response_id, _, count = name.partition(":")
        response = self._response
        if response is not None and response_id == str(response.id) and count.isdigit():
            response.played = max(response.played, int(count))
            log.debug("mark %s acknowledged", name)
            response.marked.set()

    async def _on_partial(self, text: str) -> None:
        if not text.strip():
            return
        now = time.monotonic()
        self._last_partial_at = now
        self._last_activity = now
        letters = sum(ch.isalnum() for ch in text)
        if self.config.barge_in and self._responding() and letters >= MIN_BARGE_IN_CHARS:
            log.info("barge-in: caller interrupted the agent")
            await self._cancel_response()
            await self.media.clear()

    async def _on_committed(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        now = time.monotonic()
        self._last_activity = now
        if self.config.log_transcripts:
            log.info("caller: %s", text)
        endpointing_ms = _ms(now - self._last_partial_at) if self._last_partial_at else None
        self._last_partial_at = None
        self._pending_user = _join(self._pending_user, text)
        if self._responding():
            if not self.config.barge_in:
                return  # answered once the current response finishes
            await self._cancel_response()
            await self.media.clear()
        self._start_response(user_text=self._pending_user, committed_at=now, endpointing_ms=endpointing_ms)

    # ---- responses ----------------------------------------------------------------------

    def _responding(self) -> bool:
        return self._response_task is not None and not self._response_task.done()

    def _start_response(self, *, user_text: str | None, committed_at: float, endpointing_ms: int | None) -> None:
        self._pending_user = ""
        self._next_response_id += 1
        self._response = _Response(self._next_response_id, user_text, committed_at, endpointing_ms)
        self._response_task = asyncio.create_task(self._respond(self._response))

    async def _cancel_response(self) -> None:
        """Stop the in-flight response and keep only what the caller actually heard."""
        task, response = self._response_task, self._response
        if task is None or task.done() or response is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        self._finish_turn(response, interrupted=True)

    async def _respond(self, response: _Response) -> None:
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        if response.user_text is None:
            producer = None
            sentences.put_nowait(self.config.greeting)
            sentences.put_nowait(None)
        else:
            messages = [*self.history, {"role": "user", "content": response.user_text}]
            producer = asyncio.create_task(self._generate(response, messages, sentences))
        try:
            spoken = ""
            while (sentence := await sentences.get()) is not None:
                await self._speak(response, sentence, previous_text=spoken)
                spoken = _join(spoken, sentence)
            if producer is not None:
                await producer  # surface LLM errors
            await self._await_playback(response)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("response %s failed", response.id)
            spoken = await self._apologise(response)
            self._finish_turn(response, interrupted=not spoken)
            return
        finally:
            if producer is not None:
                producer.cancel()
        self._finish_turn(response, interrupted=False)
        self._last_activity = time.monotonic()
        if response.end_call:
            self._hangup.set()
        elif self._pending_user:
            self._start_response(user_text=self._pending_user, committed_at=time.monotonic(), endpointing_ms=None)

    async def _apologise(self, response: _Response) -> bool:
        """Tell the caller something went wrong; returns whether the phrase was spoken."""
        try:
            await self._speak(response, self.config.fallback, previous_text="")
            await self._await_playback(response)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("could not speak the fallback phrase")
            return False
        return True

    async def _generate(
        self, response: _Response, messages: list[dict[str, str]], sentences: asyncio.Queue[str | None]
    ) -> None:
        splitter = SentenceSplitter()

        def emit(pieces: list[str]) -> None:
            for piece in pieces:
                if END_CALL_TOKEN in piece:
                    response.end_call = True
                    piece = piece.replace(END_CALL_TOKEN, "").strip()
                if piece:
                    sentences.put_nowait(piece)

        try:
            async for item in self.llm.stream(messages, self.config.llm_model):
                if isinstance(item, LLMUsage):
                    self.llm_usage += item
                    continue
                if response.llm_ttft_ms is None:
                    response.llm_ttft_ms = _ms(time.monotonic() - response.committed_at)
                emit(splitter.push(item))
            emit(splitter.finish())
        finally:
            sentences.put_nowait(None)

    async def _speak(self, response: _Response, sentence: str, previous_text: str) -> None:
        requested_at = time.monotonic()
        engines = [(self.tts, self.config.voice)]
        if self.fallback_tts is not None and self.config.fallback_voice is not None:
            engines.append((self.fallback_tts, self.config.fallback_voice))
        for attempt, (tts, voice) in enumerate(engines):
            produced = False
            try:
                async for pcm in tts.stream(
                    sentence,
                    voice_id=voice.voice_id,
                    model_id=voice.model_id,
                    language=self.config.language,
                    sample_rate=self.media.sample_rate,
                    previous_text=previous_text,
                ):
                    if not produced:
                        produced = True
                        if attempt == 0:
                            self.tts_characters += len(sentence)
                        else:
                            self.fallback_tts_characters += len(sentence)
                        if response.tts_ttfb_ms is None:
                            now = time.monotonic()
                            response.tts_ttfb_ms = _ms(now - requested_at)
                            if response.user_text is not None:
                                response.response_latency_ms = _ms(now - response.committed_at)
                    await self.media.send_audio(pcm)
                break
            except Exception:
                # Falling back after audio has started would speak the sentence twice.
                if produced or attempt == len(engines) - 1:
                    raise
                log.warning("TTS provider failed before producing audio; trying the fallback", exc_info=True)
        await self.media.flush_audio()
        response.sent.append(sentence)
        await self.media.mark(f"{response.id}:{len(response.sent)}")

    async def _await_playback(self, response: _Response) -> None:
        try:
            async with asyncio.timeout(self.media.playback_remaining() + MARK_GRACE_SECONDS):
                while response.played < len(response.sent):
                    await response.marked.wait()
                    response.marked.clear()
        except TimeoutError:
            log.warning("playback marks missing for response %s; assuming audio played", response.id)
            response.played = len(response.sent)

    def _finish_turn(self, response: _Response, *, interrupted: bool) -> None:
        if self._response is response:
            self._response = None
        heard = response.sent[: response.played] if interrupted else response.sent
        log.debug(
            "response %s done: interrupted=%s sent=%d played=%d",
            response.id,
            interrupted,
            len(response.sent),
            response.played,
        )
        agent_text = " ".join(heard)
        if not agent_text:
            # The caller heard nothing, so their words are still unanswered.
            if response.user_text:
                self._pending_user = _join(response.user_text, self._pending_user)
            return
        if response.user_text:
            self.history.append({"role": "user", "content": response.user_text})
        self.history.append({"role": "assistant", "content": agent_text + ("…" if interrupted else "")})
        if self.config.log_transcripts:
            log.info("agent%s: %s", " (interrupted)" if interrupted else "", agent_text)
        self.turns.append(
            {
                "idx": len(self.turns),
                "user_text": response.user_text,
                "agent_text": agent_text,
                "interrupted": interrupted,
                "endpointing_ms": response.endpointing_ms,
                "llm_ttft_ms": response.llm_ttft_ms,
                "tts_ttfb_ms": response.tts_ttfb_ms,
                "response_latency_ms": response.response_latency_ms,
            }
        )
