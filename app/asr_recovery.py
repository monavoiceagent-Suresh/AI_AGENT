"""One call's speech recognition, kept alive across a rejected keyterm request or a provider failure.

`RecoveringTranscriber` is what VoiceSession talks to. Underneath, it holds one provider session at a time and,
when that session fails (at the handshake, by an error message after connecting, or by the stream closing),
moves along a fixed plan of at most four configurations, each tried at most once per call:

    primary + keyterms  ->  primary without keyterms  ->  fallback + keyterms  ->  fallback without keyterms

- A failure that is the provider refusing the request (handshake HTTP 400/413/414/422, Scribe's
  `invalid_request`, a Deepgram error naming keyterms) while keyterms were sent goes to the same provider
  without keyterms: optional recognition hints can never by themselves end a call.
- Any other failure (an outage, auth, quota, a closed stream) goes to the fallback provider.
- Configurations without keyterms are used directly when there are none. When the plan is used up, the original
  ASRError reaches VoiceSession and the call ends as "asr_failed", exactly as before this layer existed.

Audio continuity. Everything the caller said since the last committed transcript is kept (bounded by
REPLAY_SECONDS) and replayed, in small chunks, into the replacement session before live audio resumes; audio
arriving during the switch is queued behind it. Committed transcripts are never replayed (the buffer is cleared
at each commit), so a turn is not answered twice. Limits of this approach, by design:
- An utterance longer than REPLAY_SECONDS loses its oldest audio (counted and logged as dropped_ms).
- Audio the caller spoke between the provider's commit and that commit reaching the engine (about one network
  round trip, normally silence after the utterance) is not replayed.
- Provider-side context (Scribe's previous-text conditioning) starts afresh in the new session.
VoiceSession's state (history, turns, the response in progress) is untouched: only its transcriber changes.
"""

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any

from app import logctx
from app.interfaces import ASR, CommittedTranscript, PartialTranscript, Transcriber
from app.providers.scribe import ASRError

log = logging.getLogger(__name__)

# Audio since the last committed transcript kept for replay. One utterance is normally well under this.
REPLAY_SECONDS = 8.0
# Replay chunk length: comparable to live telephony frames, so no provider sees an oversized message.
REPLAY_CHUNK_SECONDS = 0.1

# HTTP statuses with which a provider refuses the handshake request itself (bad or oversized parameters), as
# opposed to an outage or an auth failure.
_REQUEST_REFUSED = frozenset({400, 413, 414, 422})


def request_refused(exc: BaseException) -> bool:
    """The provider refused this session's configuration, at the handshake or by an error message."""
    if isinstance(exc, ASRError):
        return exc.rejected_request
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status in _REQUEST_REFUSED


def _reason(exc: BaseException) -> str:
    """A short, data-free reason: the provider's error code or HTTP status, else the exception type."""
    if isinstance(exc, ASRError) and exc.code:
        return exc.code
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return f"http_{status}" if status is not None else type(exc).__name__


@dataclass(frozen=True)
class Leg:
    """One provider this call may use, and the keyterms prepared for that provider's own limits."""

    asr: ASR
    provider: str
    role: str  # "primary" | "fallback"
    keyterms: tuple[str, ...] = ()


@dataclass
class Segment:
    """One provider session's usage: what was sent to it, for cost and the call record."""

    provider: str
    role: str
    keyterms_sent: int
    keyterms_accepted: int | None = None  # None: the provider did not (or could not) confirm
    audio_bytes: int = 0
    replayed_bytes: int = 0
    end: str = "open"  # "open" | "closed" | the failure reason that ended it

    def record(self, sample_rate: int) -> dict[str, Any]:
        per_second = 2 * sample_rate
        return {
            "provider": self.provider,
            "role": self.role,
            "seconds": round(self.audio_bytes / per_second, 2),
            "replayed_seconds": round(self.replayed_bytes / per_second, 2),
            "keyterms_sent": self.keyterms_sent,
            "keyterms_accepted": self.keyterms_accepted,
            "end": self.end,
        }


@dataclass
class _Step:
    leg: int  # index into legs
    with_keyterms: bool


class RecoveringTranscriber:
    def __init__(self, legs: Sequence[Leg], *, language: str, sample_rate: int):
        if not legs:
            raise ValueError("at least one ASR provider is required")
        self._legs = list(legs)
        self._language = language
        self._sample_rate = sample_rate
        self._tried: set[tuple[int, bool]] = set()
        self._step: _Step | None = None
        self._ctx: AbstractAsyncContextManager[Transcriber] | None = None
        self._inner: Transcriber | None = None
        self.segments: list[Segment] = []
        # Audio since the last committed transcript (the replay source), and audio waiting for a new session.
        self._since_commit = bytearray()
        self._dropped_bytes = 0
        self._pending = bytearray()
        self._paused = True  # no live session to send to (connecting, replaying or broken)
        self._replaying = False
        self._send_failure: BaseException | None = None
        self._closed = False
        self.recoveries = 0  # mid-call recoveries
        self.failures = 0  # every failed configuration, at connect time or mid-call
        self.outcome = "none"  # "none" | "recovered" | "failed"

    # ---- the Transcriber protocol ------------------------------------------------------------------------

    async def send_audio(self, pcm: bytes) -> None:
        self._remember(pcm)
        if self._paused:
            if self._replaying:
                self._pending.extend(pcm)
            return  # otherwise it is in _since_commit and will be replayed into the next session
        try:
            await self._inner.send_audio(pcm)  # type: ignore[union-attr]
        except Exception as exc:
            # This audio is already kept for replay. Closing the session ends its event stream, so events()
            # recovers even if the provider's stream would otherwise stay open.
            self._paused = True
            self._send_failure = exc
            log.warning(
                "%s asr_event=send_failed provider=%s reason=%s", self._ctx_fields(), self._provider(), _reason(exc)
            )
            await self._close_session(_reason(exc))
            return
        self.segments[-1].audio_bytes += len(pcm)

    async def events(self) -> AsyncIterator[PartialTranscript | CommittedTranscript]:
        while True:
            try:
                async for event in self._inner.events():  # type: ignore[union-attr]
                    if isinstance(event, CommittedTranscript):
                        # Never replay what has already been committed: that would answer it twice.
                        self._since_commit.clear()
                        self._dropped_bytes = 0
                    yield event
                if self._closed:
                    return
                cause: BaseException = self._send_failure or ASRError(
                    "speech recognition stream ended unexpectedly", code="stream_closed"
                )
            except Exception as exc:
                if self._closed:
                    return
                cause = self._send_failure or exc
            self._send_failure = None
            await self._recover(cause)

    @property
    def session(self) -> Transcriber | None:
        """The provider session currently in use (None between sessions)."""
        return self._inner

    # ---- lifecycle ----------------------------------------------------------------------------------------

    async def open(self) -> None:
        first = _Step(0, bool(self._legs[0].keyterms))
        await self._run_plan(first, cause=None)

    async def close(self) -> None:
        self._closed = True
        self._paused = True
        await self._close_session("closed")

    def usage(self) -> dict[str, Any]:
        """For the call record. `asr_provider` is the provider that served the end of the call."""
        self._capture_acceptance()
        records = [s.record(self._sample_rate) for s in self.segments]
        return {
            "asr_provider": self.segments[-1].provider if self.segments else "",
            "asr_keyterms_sent": any(s.keyterms_sent for s in self.segments),
            "asr_keyterms_accepted": self._accepted_summary(),
            "asr_recovery": self.outcome,
            "asr_segments": records,
        }

    # ---- recovery -----------------------------------------------------------------------------------------

    def _next_step(self, failed: _Step, cause: BaseException) -> _Step | None:
        if failed.with_keyterms and request_refused(cause):
            same = _Step(failed.leg, False)
            if (same.leg, False) not in self._tried:
                return same
        for index in range(failed.leg + 1, len(self._legs)):
            for with_keyterms in (True, False):
                if with_keyterms and not self._legs[index].keyterms:
                    continue
                if (index, with_keyterms) not in self._tried:
                    return _Step(index, with_keyterms)
        return None

    async def _recover(self, cause: BaseException) -> None:
        self._paused = True
        failed = self._step
        assert failed is not None
        self._log_failure(failed, cause, mid_call=True)
        await self._close_session(_reason(cause))
        self.recoveries += 1
        step = self._next_step(failed, cause)
        if step is None:
            self._give_up(cause)
        await self._run_plan(step, cause=cause)

    async def _run_plan(self, step: _Step | None, *, cause: BaseException | None) -> None:
        """Connect `step`, then each next step after a failure, until one session is live (and replayed into)."""
        while step is not None:
            self._tried.add((step.leg, step.with_keyterms))
            leg = self._legs[step.leg]
            if cause is not None:
                self._log_switch(self._step, step)
            self._step = step
            try:
                await self._open_session(leg, step.with_keyterms)
                await self._replay(leg)
            except Exception as exc:
                self._log_failure(step, exc, mid_call=False)
                await self._close_session(_reason(exc))
                cause = exc
                step = self._next_step(step, exc)
                continue
            if self.failures:
                self.outcome = "recovered"
                log.info(
                    "%s asr_event=recovery_outcome outcome=recovered provider=%s role=%s keyterms=%s recoveries=%d",
                    self._ctx_fields(),
                    leg.provider,
                    leg.role,
                    "on" if step.with_keyterms else "off",
                    self.recoveries,
                )
            return
        assert cause is not None
        self._give_up(cause)

    def _give_up(self, cause: BaseException) -> None:
        self.outcome = "failed"
        log.error(
            "%s asr_event=recovery_outcome outcome=failed reason=%s configurations_tried=%d",
            self._ctx_fields(),
            _reason(cause),
            len(self._tried),
        )
        raise cause

    async def _open_session(self, leg: Leg, with_keyterms: bool) -> None:
        if with_keyterms:
            ctx = leg.asr.connect(language=self._language, sample_rate=self._sample_rate, keyterms=leg.keyterms)
        else:
            # Exactly the plain connect every call made before keyterms existed.
            ctx = leg.asr.connect(language=self._language, sample_rate=self._sample_rate)
        segment = Segment(leg.provider, leg.role, len(leg.keyterms) if with_keyterms else 0)
        self.segments.append(segment)
        self._inner = await ctx.__aenter__()
        self._ctx = ctx
        log.info(
            "%s asr_event=provider_selected provider=%s role=%s keyterms_sent=%d",
            self._ctx_fields(),
            leg.provider,
            leg.role,
            segment.keyterms_sent,
        )

    async def _replay(self, leg: Leg) -> None:
        """Send everything said since the last commit to the new session, then resume live audio."""
        self._pending = bytearray(self._since_commit)
        dropped_ms = self._ms(self._dropped_bytes)
        self._replaying = True
        replayed = 0
        chunk = max(2, int(REPLAY_CHUNK_SECONDS * self._sample_rate) * 2)
        try:
            # Audio arriving meanwhile is appended to _pending, so it follows in order; nothing awaits between
            # the last send and resuming live audio, so no frame can slip between the two.
            while self._pending:
                piece = bytes(self._pending[:chunk])
                del self._pending[:chunk]
                await self._inner.send_audio(piece)  # type: ignore[union-attr]
                replayed += len(piece)
                self.segments[-1].audio_bytes += len(piece)
                self.segments[-1].replayed_bytes += len(piece)
            self._paused = False
        except Exception:
            if self.failures:
                log.warning(
                    "%s asr_event=audio_replay result=failed provider=%s replayed_ms=%d",
                    self._ctx_fields(),
                    leg.provider,
                    self._ms(replayed),
                )
            raise
        finally:
            self._replaying = False
            self._pending = bytearray()
        if self.failures:
            log.info(
                "%s asr_event=audio_replay result=ok provider=%s replayed_ms=%d dropped_ms=%d",
                self._ctx_fields(),
                leg.provider,
                self._ms(replayed),
                dropped_ms,
            )

    async def _close_session(self, end: str) -> None:
        self._capture_acceptance()
        if self.segments and self.segments[-1].end == "open":
            self.segments[-1].end = end
        ctx, self._ctx, self._inner = self._ctx, None, None
        if ctx is not None:
            with suppress(Exception):
                await ctx.__aexit__(None, None, None)

    # ---- bookkeeping --------------------------------------------------------------------------------------

    def _remember(self, pcm: bytes) -> None:
        self._since_commit.extend(pcm)
        cap = int(REPLAY_SECONDS * self._sample_rate) * 2
        if len(self._since_commit) > cap:
            excess = len(self._since_commit) - cap
            del self._since_commit[:excess]
            self._dropped_bytes += excess

    def _capture_acceptance(self) -> None:
        if self._inner is not None and self.segments:
            accepted = getattr(self._inner, "keyterms_accepted", None)
            if isinstance(accepted, int):
                self.segments[-1].keyterms_accepted = accepted

    def _accepted_summary(self) -> int | None:
        """Keyterms the provider confirmed for the last session that sent any; None when unconfirmed."""
        for segment in reversed(self.segments):
            if segment.keyterms_sent:
                return segment.keyterms_accepted
        return None

    def _provider(self) -> str:
        return self._legs[self._step.leg].provider if self._step else "-"

    def _ms(self, size: int) -> int:
        return round(size / (2 * self._sample_rate) * 1000)

    @staticmethod
    def _ctx_fields() -> str:
        return logctx.fields() or "call_id=-"

    def _log_failure(self, step: _Step, cause: BaseException, *, mid_call: bool) -> None:
        self.failures += 1
        leg = self._legs[step.leg]
        if step.with_keyterms and request_refused(cause):
            log.warning(
                "%s asr_event=keyterms_rejected provider=%s role=%s reason=%s keyterms_sent=%d phase=%s",
                self._ctx_fields(),
                leg.provider,
                leg.role,
                _reason(cause),
                len(leg.keyterms),
                "session" if mid_call else "connect",
            )
        else:
            log.warning(
                "%s asr_event=provider_failed provider=%s role=%s reason=%s phase=%s",
                self._ctx_fields(),
                leg.provider,
                leg.role,
                _reason(cause),
                "session" if mid_call else "connect",
            )

    def _log_switch(self, before: _Step | None, after: _Step) -> None:
        leg = self._legs[after.leg]
        if before is not None and before.leg == after.leg:
            log.warning(
                "%s asr_event=reconnect_without_keyterms provider=%s role=%s",
                self._ctx_fields(),
                leg.provider,
                leg.role,
            )
        else:
            log.warning(
                "%s asr_event=fallback_activated from=%s to=%s keyterms=%s",
                self._ctx_fields(),
                self._legs[before.leg].provider if before else "-",
                leg.provider,
                "on" if after.with_keyterms else "off",
            )


@asynccontextmanager
async def recovering_asr(transcriber: RecoveringTranscriber) -> AsyncIterator[RecoveringTranscriber]:
    """Open the first working configuration of the transcriber's plan; raises the last failure if none connects.
    The transcriber keeps its usage (segments, outcome) either way, for the call record."""
    await transcriber.open()
    try:
        yield transcriber
    finally:
        await transcriber.close()
