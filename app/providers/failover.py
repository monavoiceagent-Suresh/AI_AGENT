"""Primary model with an automatic fallback model, both inside one response deadline.

Production runs OpenRouter only, so primary and fallback are normally the SAME OpenRouter client asked for two
different models (e.g. google/gemini-2.5-flash-lite, then openai/gpt-4o-mini). The fallback is taken only when the
primary fails before it has produced any text: timeouts, connection errors, rate limits (including a 429 cooldown
or the self-throttle refusing locally), server errors, stream errors and model-specific client errors. It is never
taken after speaking has started, because a second generation would make the caller hear the same words again,
and never for an account error (bad key, no credits), which every model on the same key shares.

OpenRouter's own `models: [...]` request fallback was considered and not used: its documentation covers rate
limits, 5xx and moderation, but not timeouts or `stream: true` explicitly, and it hides which attempt failed from
our logs. Doing it here keeps one deadline over both attempts and logs which model served each response.
"""

import inspect
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from app import logctx
from app.interfaces import LLM, LLMServed, LLMUsage
from app.providers.chat_completions import LLMError, _reason, is_model_specific

log = logging.getLogger(__name__)

# Long enough for a slow-but-healthy first token, short enough that a caller is never left in silence: once it
# passes, the session speaks its fallback phrase instead of waiting further.
DEFAULT_RESPONSE_DEADLINE_SECONDS = 8.0
# The most of the response deadline the primary model may use when a fallback model is waiting behind it.
PRIMARY_SHARE_OF_DEADLINE = 0.5


def _describe(exc: BaseException) -> str:
    """Name the failure without its message: provider bodies can echo prompts or credentials."""
    status = getattr(exc, "status", None)
    return type(exc).__name__ + (f" {status}" if status else "")


class LLMUnavailable(LLMError):
    """Neither the primary nor the fallback model could answer. `reason` is log/DB-safe, e.g.
    "primary=rate_limited fallback=timeout"."""

    def __init__(self, message: str, *, status: int | None, reason: str):
        super().__init__(message, status=status)
        self.reason = reason


def _accepts_overrides(llm: Any) -> bool:
    try:
        return "deadline" in inspect.signature(llm.stream).parameters
    except (TypeError, ValueError):
        return False


class FailoverLLM:
    def __init__(
        self,
        *,
        primary: LLM,
        fallback: LLM | None,
        fallback_model: str,
        primary_timeout_s: float | None = None,
        primary_retries: int | None = None,
        fallback_timeout_s: float | None = None,
        fallback_retries: int | None = None,
        deadline_s: float = DEFAULT_RESPONSE_DEADLINE_SECONDS,
    ):
        """`fallback` is None (or `fallback_model` empty) when no fallback is configured; then the primary's errors
        are raised. The timeout/retry overrides apply only to clients that accept them (ChatCompletionsLLM)."""
        self._primary = primary
        self._fallback = fallback if fallback_model else None
        self._fallback_model = fallback_model
        if self._fallback is not None:
            # With a fallback model behind it, the primary must leave the fallback real time to answer: the
            # fallback IS its retry, so the primary never retries itself, and a slow primary is cut off early
            # enough (at most half the deadline) for the fallback to still reply inside the deadline. Without
            # this, a long OPENROUTER_TIMEOUT_MS (or retries) on the primary would use the whole deadline.
            primary_retries = 0
            cap = deadline_s * PRIMARY_SHARE_OF_DEADLINE
            primary_timeout_s = min(primary_timeout_s, cap) if primary_timeout_s else cap
        self._primary_opts = {"timeout_s": primary_timeout_s, "max_retries": primary_retries, "role": "primary"}
        self._fallback_opts = {"timeout_s": fallback_timeout_s, "max_retries": fallback_retries, "role": "fallback"}
        self._deadline_s = deadline_s

    @property
    def fallback_model(self) -> str | None:
        return self._fallback_model if self._fallback is not None else None

    def _open(self, llm: LLM, messages, model: str, opts: dict, deadline: float):
        if _accepts_overrides(llm):
            return llm.stream(messages, model, deadline=deadline, **opts)
        return llm.stream(messages, model)

    async def stream(self, messages: list[dict[str, str]], model: str) -> AsyncIterator[str | LLMUsage | LLMServed]:
        deadline = time.monotonic() + self._deadline_s
        plan: list[tuple[LLM, str, dict]] = [(self._primary, model, self._primary_opts)]
        if self._fallback is not None and self._fallback_model != model:
            plan.append((self._fallback, self._fallback_model, self._fallback_opts))
        failures: list[str] = []
        last_exc: BaseException | None = None
        for index, (llm, attempt_model, opts) in enumerate(plan):
            produced = False
            try:
                async for item in self._open(llm, messages, attempt_model, opts, deadline):
                    if isinstance(item, LLMServed):
                        item = LLMServed(provider=item.provider, model=item.model, role=opts["role"])
                        if opts["role"] == "fallback":
                            log.warning(
                                "llm fallback served %s primary_model=%s fallback_model=%s primary_failure=%s",
                                logctx.fields() or "call_id=-",
                                model,
                                item.model,
                                ",".join(failures) or "-",
                            )
                    elif isinstance(item, str):
                        produced = True
                    yield item
                return
            except Exception as exc:
                failures.append(f"{opts['role']}={_reason(exc)}")
                last_exc = exc
                last_step = index == len(plan) - 1
                if produced or not is_model_specific(exc) or last_step:
                    if not produced and len(plan) > 1 and last_step:
                        raise LLMUnavailable(
                            f"no LLM could answer ({', '.join(failures)})",
                            status=getattr(exc, "status", None),
                            reason=" ".join(failures),
                        ) from exc
                    raise
                log.warning(
                    "primary LLM failed before replying (%s) %s model=%s; trying fallback model %s",
                    _describe(exc),
                    logctx.fields() or "call_id=-",
                    attempt_model,
                    self._fallback_model,
                )
        if last_exc is not None:  # pragma: no cover - the loop always returns or raises
            raise last_exc

    async def complete(self, messages: list[dict[str, str]], model: str) -> tuple[str, LLMUsage]:
        if self._fallback is None or self._fallback_model == model:
            return await self._primary.complete(messages, model)
        try:
            return await self._primary.complete(messages, model)
        except Exception as exc:
            if not is_model_specific(exc):
                raise
            log.warning("primary LLM failed (%s); using fallback model %s", _describe(exc), self._fallback_model)
            return await self._fallback.complete(messages, self._fallback_model)
