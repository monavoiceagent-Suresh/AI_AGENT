"""Voice-session orchestrator: caller audio -> ASR -> LLM -> TTS -> caller, with barge-in.

Delivery is tracked in three distinct stages, never conflated:
- synthesized: the TTS provider returned audio for a sentence;
- sent: that audio (and its playback mark) was written to the Exotel media stream;
- confirmed: Exotel echoed the sentence's mark back, i.e. reported it as played.
A sentence whose mark never comes back is recorded as sent but unconfirmed, not as played.

What the caller can currently hear is judged by the media leg's own playback clock (`playback_remaining`), so
barge-in applies only while agent audio is actually audible, and a caller replying after the agent finished
is never mistaken for an interruption just because the last mark has not arrived yet.
"""

import asyncio
import logging
import re
import time
import unicodedata
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from app import logctx
from app.interfaces import (
    LLM,
    TTS,
    CallAudio,
    CallEnded,
    CallMedia,
    CommittedTranscript,
    LLMServed,
    LLMUsage,
    PlaybackMark,
    Transcriber,
    TTSError,
    Voice,
)
from app.prompts import END_CALL_TOKEN

log = logging.getLogger(__name__)

SENTENCE_END = re.compile(r"[.!?।॥](?=\s)|\n")
CLAUSE_END = re.compile(r"[,;:](?=\s)")
# Start speaking at a clause boundary once this much text is waiting, rather than wait for a full stop.
SOFT_FLUSH_CHARS = 60
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


def speech_chars(text: str) -> int:
    """Letters, digits and combining marks in any script: Devanagari/Tamil vowel signs count, so a short Hindi
    or Tamil word is measured like an English word of the same spoken length. Punctuation and spaces don't."""
    return sum(1 for ch in text if unicodedata.category(ch)[0] in "LNM")


class AudioSendError(Exception):
    """Synthesized audio, a flush or a playback mark could not be written to the caller's media stream."""


class ASRFailure(Exception):
    """The speech recogniser stopped accepting audio mid-call."""


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
    # A committed transcript identical to the one just answered, delivered again within this window, is a
    # redelivery (ASR/websocket retransmit), not a new turn: 0 disables the check.
    llm_dedup_window_ms: int = 0
    # Ends the call instead of trying forever once this many responses in a row have failed.
    max_consecutive_llm_failures: int = 3
    # Barge-in needs this much caller speech during audible agent playback (see Settings).
    barge_in_min_chars: int = 4
    barge_in_min_speech_ms: int = 400
    # The greeting cannot be interrupted until its audio has played this long.
    greeting_protect_ms: int = 2500
    # Spoken once after this many seconds of caller silence, before idle_timeout_seconds ends the call
    # outright. 0 disables the nudge.
    silence_nudge_seconds: float = 0
    silence_nudge_phrase: str = ""


@dataclass
class _Response:
    id: int
    user_text: str | None
    committed_at: float
    endpointing_ms: int | None
    sent: list[str] = field(default_factory=list)  # sentences whose audio and mark were sent
    sent_ends: list[float] = field(default_factory=list)  # playback-clock time each sent sentence finishes
    played: int = 0  # sentences confirmed played via marks
    marked: asyncio.Event = field(default_factory=asyncio.Event)
    end_call: bool = False
    audio_started_at: float | None = None
    draining: bool = False  # everything is sent; only waiting for playback to finish/confirm
    is_nudge: bool = False  # a silence check-in (_watchdog), not a greeting or an LLM-driven reply
    llm_model: str | None = None
    llm_ttft_ms: int | None = None
    tts_ttfb_ms: int | None = None
    response_latency_ms: int | None = None

    @property
    def is_greeting(self) -> bool:
        return self.user_text is None


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


def _join(*parts: str) -> str:
    return " ".join(p for p in parts if p).strip()


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def failure_reason(exc: BaseException) -> str:
    """A short, log- and DB-safe category for a failed response; never a provider message body."""
    from app.providers.budget import CallBudgetExceeded
    from app.providers.chat_completions import LLMError, _reason
    from app.providers.failover import LLMUnavailable

    if isinstance(exc, LLMUnavailable):
        return f"llm_unavailable({exc.reason})"
    if isinstance(exc, CallBudgetExceeded):
        return "llm_call_budget_exceeded"
    if isinstance(exc, LLMError):
        return f"llm_failed({_reason(exc)})"
    if isinstance(exc, AudioSendError):
        return "audio_send_failed"
    if isinstance(exc, TTSError):
        return "tts_failed"
    return f"error({type(exc).__name__})"


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
        call_id: int | None = None,
        call_sid: str = "",
        tts_provider: str = "",
        fallback_tts_provider: str = "",
    ):
        self.config = config
        self.media = media
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.fallback_tts = fallback_tts
        # For correlating logs across the pipeline; cosmetic only, never used for control flow.
        self.call_id = call_id
        self.call_sid = call_sid
        self._tts_names = (tts_provider or type(tts).__name__, fallback_tts_provider or "fallback")
        self.history: list[dict[str, str]] = [{"role": "system", "content": config.system_prompt}]
        self.turns: list[dict[str, Any]] = []
        # Every response that failed, in order, as (response id, reason): persisted with the call's usage.
        self.failures: list[dict[str, Any]] = []
        self.asr_bytes = 0
        self.tts_characters = 0
        self.fallback_tts_characters = 0
        self.llm_usage = LLMUsage(0, 0, 0.0)
        self._pending_user = ""
        self._response: _Response | None = None
        self._response_task: asyncio.Task | None = None
        self._next_response_id = 0
        self._last_partial_at: float | None = None
        self._speech_started_at: float | None = None
        self._last_activity = time.monotonic()
        self._nudged = False
        self._hangup = asyncio.Event()
        self._hangup_reason = "agent_ended"
        # Guards against the LLM call paths this module controls: a redelivered commit of the same utterance,
        # and a run of failed responses that would otherwise keep retrying for the rest of the call.
        self._last_dispatched_text = ""
        self._last_dispatched_at = 0.0
        self._consecutive_llm_failures = 0

    @property
    def asr_seconds(self) -> float:
        return self.asr_bytes / (2 * self.media.sample_rate)

    def _ctx(self, response_id: int | None = None) -> str:
        """call_id/call_sid/stream_sid/response_id as a log prefix, for correlating one failure across
        the whole pipeline (LLM -> TTS -> media send)."""
        stream_sid = getattr(self.media, "stream_sid", "") or "-"
        parts = [f"call_id={self.call_id or '-'}", f"call_sid={self.call_sid or '-'}", f"stream_sid={stream_sid}"]
        if response_id is not None:
            parts.append(f"response_id={response_id}")
        return " ".join(parts)

    async def run(self) -> str:
        """Converse until the call ends; returns the terminal reason (deterministic, never empty)."""
        logctx.set_call(
            call_id=self.call_id, call_sid=self.call_sid, stream_sid=getattr(self.media, "stream_sid", "") or ""
        )
        self._start_response(user_text=None, committed_at=time.monotonic(), endpointing_ms=None)
        # Listed in priority order: if two finish in the same loop iteration, the earlier one names the end.
        named = [
            ("caller", asyncio.create_task(self._pump_call())),
            ("agent", asyncio.create_task(self._wait_hangup())),
            ("asr", asyncio.create_task(self._pump_asr())),
            ("watchdog", asyncio.create_task(self._watchdog())),
        ]
        await asyncio.wait([task for _, task in named], return_when=asyncio.FIRST_COMPLETED)
        for _, task in named:
            if not task.done():
                task.cancel()
        await asyncio.gather(*(task for _, task in named), return_exceptions=True)
        await self._cancel_response(interrupted_by="call_ended")
        name, finished = next((n, t) for n, t in named if t.done() and not t.cancelled())
        exc = finished.exception()
        if exc is None:
            reason = finished.result()
        elif isinstance(exc, ASRFailure) or name == "asr":
            reason = f"asr_failed:{type(exc.__cause__ or exc).__name__}"
        elif name == "caller":
            reason = f"media_failed:{type(exc).__name__}"
        else:
            reason = f"error:{type(exc).__name__}"
        if exc is not None:
            log.error("%s: voice session ended by a failure: %s", self._ctx(), reason, exc_info=exc)
        else:
            log.info("%s: voice session ended: %s", self._ctx(), reason)
        return reason

    # ---- inputs -------------------------------------------------------------------------

    async def _pump_call(self) -> str:
        async for event in self.media.events():
            if isinstance(event, CallAudio):
                self.asr_bytes += len(event.pcm)
                try:
                    await self.asr.send_audio(event.pcm)
                except Exception as exc:
                    raise ASRFailure("speech recogniser stopped accepting audio") from exc
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
            if (
                self.config.silence_nudge_seconds
                and not self._nudged
                and not self._responding()
                and not self._pending_user
                and now - self._last_activity >= self.config.silence_nudge_seconds
            ):
                self._nudged = True
                self._start_response(user_text="", committed_at=now, endpointing_ms=None, is_nudge=True)
            if not self._responding() and now - self._last_activity >= self.config.idle_timeout_seconds:
                return "idle_timeout"

    async def _wait_hangup(self) -> str:
        await self._hangup.wait()
        return self._hangup_reason

    def _end_call(self, reason: str) -> None:
        self._hangup_reason = reason
        self._hangup.set()

    def _on_mark(self, name: str) -> None:
        response_id, _, count = name.partition(":")
        response = self._response
        if response is not None and response_id == str(response.id) and count.isdigit():
            response.played = max(response.played, int(count))
            log.debug("%s: mark %s confirmed played", self._ctx(response.id), name)
            response.marked.set()

    # ---- what the caller can hear ---------------------------------------------------------

    def _audible(self) -> bool:
        """Whether agent audio is playing to the caller right now, by the media leg's playback clock."""
        response = self._response
        return (
            response is not None
            and response.audio_started_at is not None
            and self.media.playback_remaining() > 0
        )

    def _greeting_protected(self, response: _Response, now: float) -> bool:
        if not response.is_greeting:
            return False
        if response.audio_started_at is None:
            return True
        return _ms(now - response.audio_started_at) < self.config.greeting_protect_ms

    def _meaningful(self, text: str, now: float) -> tuple[bool, int, int]:
        """Whether caller speech is enough to interrupt: (meaningful, chars, speech ms so far)."""
        chars = speech_chars(text)
        speech_ms = _ms(now - self._speech_started_at) if self._speech_started_at is not None else 0
        enough = chars >= self.config.barge_in_min_chars and speech_ms >= self.config.barge_in_min_speech_ms
        return enough, chars, speech_ms

    async def _barge_in(self, response: _Response, *, chars: int, speech_ms: int, source: str) -> None:
        log.info(
            "%s: barge-in (%s) during audible playback: chars=%d speech_ms=%d sent=%d confirmed=%d",
            self._ctx(response.id),
            source,
            chars,
            speech_ms,
            len(response.sent),
            response.played,
        )
        await self._cancel_response(interrupted_by="barge_in")
        await self.media.clear()

    async def _on_partial(self, text: str) -> None:
        if not text.strip():
            return
        now = time.monotonic()
        if self._speech_started_at is None:
            self._speech_started_at = now
        self._last_partial_at = now
        self._last_activity = now
        self._nudged = False
        response = self._response
        if not (self.config.barge_in and self._responding() and response is not None):
            return
        # Not audible (waiting for the LLM/TTS, or playback already over): the caller is not talking over the
        # agent. Their words are handled when the ASR commits them.
        if not self._audible() or self._greeting_protected(response, now):
            return
        meaningful, chars, speech_ms = self._meaningful(text, now)
        if meaningful:
            await self._barge_in(response, chars=chars, speech_ms=speech_ms, source="partial")

    async def _on_committed(self, text: str) -> None:
        text = text.strip()
        if not text:
            # An empty commit (VAD/endpointing triggered on noise or a breath, nothing transcribed) still ends
            # that speech attempt: clear its tracking now, or a later empty-then-real sequence inflates the
            # NEXT utterance's speech_ms by however long ago this one started (live: 28s on a 5-char partial).
            self._last_partial_at = None
            self._speech_started_at = None
            return
        now = time.monotonic()
        self._last_activity = now
        self._nudged = False
        if self.config.log_transcripts:
            log.info("caller: %s", text)
        if self._is_duplicate_redelivery(text, now):
            log.info("dropped a committed transcript identical to the one just answered (redelivery)")
            return
        meaningful, chars, speech_ms = self._meaningful(text, now)
        endpointing_ms = _ms(now - self._last_partial_at) if self._last_partial_at else None
        self._last_partial_at = None
        self._speech_started_at = None
        self._pending_user = _join(self._pending_user, text)
        response = self._response
        if self._responding() and response is not None:
            if response.draining and not self._audible():
                # The agent has finished speaking (by the playback clock); only its marks are outstanding. The
                # caller is replying, not interrupting: complete the turn with everything it said.
                await self._cancel_response(completed=True)
            elif not self.config.barge_in:
                return  # answered once the current response finishes
            elif response.is_greeting and (self._greeting_protected(response, now) or not self._audible()):
                log.info("%s: caller spoke during the greeting; answering after it", self._ctx(response.id))
                return
            elif self._audible():
                if not meaningful:
                    return  # a short backchannel over the agent: answered once it finishes, not an interruption
                await self._barge_in(response, chars=chars, speech_ms=speech_ms, source="commit")
            else:
                # Nothing audible yet: the caller kept talking after a pause the ASR took as the end of their turn.
                # The in-flight reply answers only half of what they said, so it is replaced by one that answers
                # all of it (the text is merged below; nothing the caller heard is lost).
                log.info("%s: response superseded by more caller speech before any audio", self._ctx(response.id))
                await self._cancel_response(interrupted_by="superseded")
                if response.sent:
                    await self.media.clear()
        if self._hangup.is_set():
            return  # the agent already said goodbye; the call is ending
        self._start_response(user_text=self._pending_user, committed_at=now, endpointing_ms=endpointing_ms)

    def _is_duplicate_redelivery(self, text: str, now: float) -> bool:
        """Whether `text` is exactly the utterance just dispatched, delivered again moments later.

        Compares only the single incoming fragment against the last whole dispatch, so it can suppress a
        resent duplicate but never a genuinely new utterance: a new one essentially never normalizes to be
        byte-identical to everything that was just answered.
        """
        window = self.config.llm_dedup_window_ms / 1000
        if not window or _normalize(text) != self._last_dispatched_text:
            return False
        return now - self._last_dispatched_at < window

    # ---- responses ----------------------------------------------------------------------

    def _responding(self) -> bool:
        return self._response_task is not None and not self._response_task.done()

    def _start_response(
        self, *, user_text: str | None, committed_at: float, endpointing_ms: int | None, is_nudge: bool = False
    ) -> None:
        self._pending_user = ""
        self._next_response_id += 1
        self._response = _Response(self._next_response_id, user_text, committed_at, endpointing_ms, is_nudge=is_nudge)
        self._response_task = asyncio.create_task(self._respond(self._response))
        if user_text:
            self._last_dispatched_text = _normalize(user_text)
            self._last_dispatched_at = committed_at

    async def _cancel_response(self, *, interrupted_by: str | None = None, completed: bool = False) -> None:
        """Stop the in-flight response. Normally keep only what the caller actually heard; with `completed`, the
        response had already finished playing, so it is recorded whole (and any end-of-call it asked for stands)."""
        task, response = self._response_task, self._response
        if task is None or task.done() or response is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        if completed:
            self._consecutive_llm_failures = 0
            self._finish_turn(response, interrupted=False)
            if response.end_call:
                log.info("%s: agent ended the call after: %r", self._ctx(response.id), " ".join(response.sent))
                self._end_call("agent_ended")
        else:
            self._finish_turn(response, interrupted=True, interrupted_by=interrupted_by)

    async def _respond(self, response: _Response) -> None:
        logctx.set_response(response.id)
        sentences: asyncio.Queue[str | None] = asyncio.Queue()
        if response.is_greeting:
            producer = None
            # Spoken sentence by sentence like any reply: the first words start sooner, and an early interruption
            # only wastes synthesis of what had already started.
            splitter = SentenceSplitter()
            for sentence in [*splitter.push(self.config.greeting), *splitter.finish()]:
                sentences.put_nowait(sentence)
            sentences.put_nowait(None)
        elif response.is_nudge:
            producer = None
            sentences.put_nowait(self.config.silence_nudge_phrase)
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
        except Exception as exc:
            reason = failure_reason(exc)
            known = reason.startswith(("llm_", "tts_", "audio_"))
            log.error(
                "%s: response failed: %s (sent=%d confirmed=%d)",
                self._ctx(response.id),
                reason,
                len(response.sent),
                response.played,
                exc_info=None if known else exc,
            )
            self.failures.append({"response_id": response.id, "reason": reason})
            spoken = await self._apologise(response)
            self._finish_turn(response, interrupted=not spoken, failure_reason=reason)
            self._consecutive_llm_failures += 1
            if self._consecutive_llm_failures >= self.config.max_consecutive_llm_failures:
                log.warning(
                    "%s: ending call after %d consecutive failed responses",
                    self._ctx(response.id),
                    self._consecutive_llm_failures,
                )
                self._end_call("response_failures")
            elif self._pending_user and not self._hangup.is_set():
                self._start_response(user_text=self._pending_user, committed_at=time.monotonic(), endpointing_ms=None)
            return
        finally:
            if producer is not None:
                producer.cancel()
        self._consecutive_llm_failures = 0
        self._finish_turn(response, interrupted=False)
        if not response.is_nudge:
            # A nudge is itself a side effect of silence: speaking it must not restart the idle clock it was
            # meant to run alongside, or the caller would never actually hit idle_timeout.
            self._last_activity = time.monotonic()
            self._nudged = False
        if response.end_call:
            log.info("%s: agent ended the call after: %r", self._ctx(response.id), " ".join(response.sent))
            self._end_call("agent_ended")
        elif self._pending_user:
            self._start_response(user_text=self._pending_user, committed_at=time.monotonic(), endpointing_ms=None)

    async def _apologise(self, response: _Response) -> bool:
        """Tell the caller something went wrong; returns whether the phrase was sent to them."""
        try:
            await self._speak(response, self.config.fallback, previous_text="")
            await self._await_playback(response)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error(
                "%s: could not speak the fallback phrase: %s", self._ctx(response.id), failure_reason(exc)
            )
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
                if isinstance(item, LLMServed):
                    response.llm_model = item.model
                    continue
                if response.llm_ttft_ms is None:
                    response.llm_ttft_ms = _ms(time.monotonic() - response.committed_at)
                emit(splitter.push(item))
            emit(splitter.finish())
        finally:
            sentences.put_nowait(None)

    async def _send(self, response: _Response, what: str, send) -> None:
        try:
            await send()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error(
                "%s: audio %s failed: %s (sent=%d confirmed=%d)",
                self._ctx(response.id),
                what,
                type(exc).__name__,
                len(response.sent),
                response.played,
            )
            raise AudioSendError(f"audio {what} failed") from exc

    async def _speak(self, response: _Response, sentence: str, previous_text: str) -> None:
        requested_at = time.monotonic()
        engines = [(self.tts, self.config.voice, self._tts_names[0])]
        if self.fallback_tts is not None and self.config.fallback_voice is not None:
            engines.append((self.fallback_tts, self.config.fallback_voice, self._tts_names[1]))
        index = len(response.sent) + 1
        for attempt, (tts, voice, name) in enumerate(engines):
            produced = False
            audio_bytes = 0
            first_byte_ms: int | None = None
            try:
                async for pcm in tts.stream(
                    sentence,
                    voice_id=voice.voice_id,
                    model_id=voice.model_id,
                    language=self.config.language,
                    sample_rate=self.media.sample_rate,
                    previous_text=previous_text,
                    voice_settings=voice.settings,
                ):
                    if not produced:
                        produced = True
                        now = time.monotonic()
                        first_byte_ms = _ms(now - requested_at)
                        if attempt == 0:
                            self.tts_characters += len(sentence)
                        else:
                            self.fallback_tts_characters += len(sentence)
                        if response.tts_ttfb_ms is None:
                            response.tts_ttfb_ms = first_byte_ms
                            if response.user_text is not None:
                                response.response_latency_ms = _ms(now - response.committed_at)
                    audio_bytes += len(pcm)
                    await self._send(response, "send", lambda pcm=pcm: self.media.send_audio(pcm))
                    if response.audio_started_at is None:
                        response.audio_started_at = time.monotonic()
                if not produced:
                    raise TTSError(f"{name} returned no audio")
                log.info(
                    "%s: tts sentence=%d provider=%s model=%s chars=%d first_byte_ms=%s synth_ms=%d audio_ms=%d",
                    self._ctx(response.id),
                    index,
                    name,
                    voice.model_id,
                    len(sentence),
                    first_byte_ms,
                    _ms(time.monotonic() - requested_at),
                    round(audio_bytes / (2 * self.media.sample_rate) * 1000),
                )
                break
            except AudioSendError:
                raise  # a transport problem: another TTS provider cannot fix it
            except Exception as exc:
                log.warning(
                    "%s: tts sentence=%d provider=%s model=%s failed %s audio: %s",
                    self._ctx(response.id),
                    index,
                    name,
                    voice.model_id,
                    "after" if produced else "before",
                    str(exc)[:200] if isinstance(exc, TTSError) else type(exc).__name__,
                )
                # Falling back after audio has started would speak the sentence twice.
                if produced or attempt == len(engines) - 1:
                    raise
        await self._send(response, "flush", self.media.flush_audio)
        response.sent.append(sentence)
        response.sent_ends.append(time.monotonic() + self.media.playback_remaining())
        await self._send(response, "mark", lambda: self.media.mark(f"{response.id}:{len(response.sent)}"))

    async def _await_playback(self, response: _Response) -> None:
        """Wait until Exotel confirms every sent sentence played, bounded by the playback clock plus a grace.
        A missing confirmation is recorded as unconfirmed playback; it is never counted as played."""
        response.draining = True
        try:
            async with asyncio.timeout(self.media.playback_remaining() + MARK_GRACE_SECONDS):
                while response.played < len(response.sent):
                    await response.marked.wait()
                    response.marked.clear()
        except TimeoutError:
            log.warning(
                "%s: playback unconfirmed: marks missing after the playback clock ran out (sent=%d confirmed=%d)",
                self._ctx(response.id),
                len(response.sent),
                response.played,
            )

    def _heard_count(self, response: _Response) -> int:
        """Sentences the caller heard of an interrupted response: those Exotel confirmed, or that the playback
        clock says had finished playing when it was interrupted, whichever is more."""
        now = time.monotonic()
        by_clock = sum(1 for end in response.sent_ends if end <= now)
        return min(len(response.sent), max(response.played, by_clock))

    def _finish_turn(
        self,
        response: _Response,
        *,
        interrupted: bool,
        interrupted_by: str | None = None,
        failure_reason: str | None = None,
    ) -> None:
        if self._response is response:
            self._response = None
        heard = response.sent[: self._heard_count(response)] if interrupted else response.sent
        confirmed = min(response.played, len(response.sent))
        playback_confirmed = confirmed >= len(response.sent) if response.sent else None
        log.info(
            "%s: turn done: interrupted=%s by=%s sent=%d confirmed=%d heard=%d playback_confirmed=%s llm_model=%s "
            "failure=%s",
            self._ctx(response.id),
            interrupted,
            interrupted_by or "-",
            len(response.sent),
            confirmed,
            len(heard),
            playback_confirmed,
            response.llm_model or "-",
            failure_reason or "-",
        )
        agent_text = " ".join(heard)
        if not agent_text:
            # The caller heard nothing, so their words are still unanswered. Expected when a barge-in or
            # cancellation cut the response off before any audio went out; worth flagging otherwise, since
            # a completed (non-interrupted) response with zero audio sent is the "LLM replied, caller
            # heard silence" failure mode.
            log_fn = log.debug if interrupted else log.warning
            log_fn("%s: caller heard no audio for this response (interrupted=%s)", self._ctx(response.id), interrupted)
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
                "sentences_sent": len(response.sent),
                "sentences_confirmed": confirmed,
                "playback_confirmed": playback_confirmed,
                "interrupted_by": interrupted_by,
                "failure_reason": failure_reason,
                "llm_model": response.llm_model,
            }
        )
