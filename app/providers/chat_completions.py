"""OpenAI-compatible chat completions with streaming. OpenRouter and Groq both speak this format.

Only `delta.content` is spoken. Reasoning models (gpt-oss) stream their thinking in a separate reasoning field
(`delta.reasoning` / `delta.reasoning_content`); it is discarded and never reaches the caller.
"""

import asyncio
import json
import logging
import random
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app import logctx
from app.interfaces import LLMServed, LLMUsage
from app.providers.rate_limit import TokenBucket

log = logging.getLogger(__name__)

# Replies must start within a few seconds, so a single retry is the most a call can afford.
COMPLETE_TIMEOUT_SECONDS = 30
# Post-call outcome extraction returns a short JSON object; it needs room for reasoning, but no more.
MIN_COMPLETE_TOKENS = 600
# A 429 with no Retry-After header: back off for this long before trying this provider again.
DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 1.0


class LLMError(Exception):
    """A provider call failed. `status` is the HTTP status, or None for a transport or stream error."""

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


def is_rate_limited(exc: BaseException) -> bool:
    return isinstance(exc, LLMError) and exc.status == 429


def is_transient(exc: BaseException) -> bool:
    """Timeouts, connection failures, rate limits, server errors and stream errors. Client errors are not transient."""
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, LLMError) and (exc.status is None or exc.status == 429 or exc.status >= 500)


# The same API key serves every model, so these fail identically on a fallback model: never worth switching for.
ACCOUNT_ERRORS = frozenset({401, 402, 403})


def is_model_specific(exc: BaseException) -> bool:
    """Whether another model on the same gateway could succeed where this one failed: every transient error, plus
    the client errors that are about this model (unknown/unavailable model, a request it rejects). Not an
    authentication, credit or permission problem, which every model on the same key shares."""
    if is_transient(exc):
        return True
    status = getattr(exc, "status", None)
    return isinstance(exc, LLMError) and status is not None and status not in ACCOUNT_ERRORS


def _status_label(exc: BaseException) -> str:
    """How an attempt's outcome is logged: the HTTP status if there is one, else the exception's type."""
    status = getattr(exc, "status", None)
    return str(status) if status is not None else type(exc).__name__


def _reason(exc: BaseException) -> str:
    """A short, log-safe failure category: never the provider's message body (it can echo prompts)."""
    if isinstance(exc, httpx.TimeoutException):
        return "timeout"
    if isinstance(exc, httpx.TransportError):
        return "connection"
    status = getattr(exc, "status", None)
    if status == 429:
        return "rate_limited"
    if isinstance(status, int) and status >= 500:
        return "upstream_error"
    if isinstance(status, int):
        return f"http_{status}"
    return "stream_error"


def _usage(payload: dict[str, Any]) -> LLMUsage:
    usage = payload.get("usage") or {}
    return LLMUsage(
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        cost_usd=float(usage.get("cost") or 0.0),
    )


class ChatCompletionsLLM:
    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        name: str,
        api_key: str,
        base_url: str,
        max_tokens: int,
        temperature: float,
        timeout_ms: int,
        max_retries: int,
        limiter: TokenBucket | None = None,
        retry_backoff_base_ms: int = 300,
        retry_backoff_max_ms: int = 2000,
    ):
        self._http = http
        self._name = name
        self._secret = api_key
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout_ms / 1000
        self._max_retries = max_retries
        # A per-process self-throttle and a short cooldown set after a 429, honouring Retry-After. Both are kept
        # per MODEL, not per provider: an upstream limit on one model (e.g. Gemini on Google AI Studio) says
        # nothing about another model behind the same gateway, so it must not stall concurrent calls that are
        # using, or failing over to, a different model. `limiter` is the template: each model gets its own bucket
        # with the same per-minute budget.
        self._limiter_template = limiter
        self._limiters: dict[str, TokenBucket] = {}
        self._backoff_base = retry_backoff_base_ms / 1000
        self._backoff_max = retry_backoff_max_ms / 1000
        self._cooldown_until: dict[str, float] = {}

    def _body(self, messages: list[dict[str, str]], model: str, stream: bool, max_tokens: int) -> dict[str, Any]:
        return {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self._temperature,
            "stream": stream,
        }

    def _redact(self, text: str) -> str:
        return text.replace(self._secret, "***")

    def _retry_after_seconds(self, response: httpx.Response) -> float:
        value = response.headers.get("retry-after")
        if value is None:
            return DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS
        try:
            return max(0.0, float(value))
        except ValueError:
            return DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS

    def _http_error(self, response: httpx.Response, body: bytes, model: str) -> LLMError:
        if response.status_code == 429:
            # Stop hitting a model that just told us to slow down; the next attempt for this model (a retry, or a
            # concurrent call's turn) finds the cooldown still active and goes to its fallback instead.
            self._cooldown_until[model] = time.monotonic() + self._retry_after_seconds(response)
        detail = self._redact(body[:300].decode("utf-8", "replace"))
        return LLMError(f"{self._name} failed ({response.status_code}): {detail}", status=response.status_code)

    def cooldown_remaining(self, model: str) -> float:
        return max(0.0, self._cooldown_until.get(model, 0.0) - time.monotonic())

    def _limiter_for(self, model: str) -> TokenBucket | None:
        template = self._limiter_template
        if template is None:
            return None
        if model not in self._limiters:
            self._limiters[model] = TokenBucket(template.max_per_minute)
        return self._limiters[model]

    def _check_budget(self, model: str) -> None:
        """Raise before making a request this model's own limits say we shouldn't make yet."""
        if self.cooldown_remaining(model) > 0:
            raise LLMError(f"{self._name} {model} is cooling down after a 429", status=429)
        limiter = self._limiter_for(model)
        if limiter is not None and not limiter.try_acquire():
            raise LLMError(f"{self._name} {model} self-throttled: per-minute request budget reached", status=429)

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter before a retry: attempt 0 waits ~base, attempt 1 ~2x base, capped."""
        delay = min(self._backoff_max, self._backoff_base * (2**attempt))
        return delay + random.uniform(0, self._backoff_base)

    def _log(
        self,
        *,
        request_id: str,
        model: str,
        attempt: int,
        status: str,
        started: float,
        level: int = logging.INFO,
        **extra: Any,
    ) -> None:
        # Never includes messages, response bodies or the API key: only identifiers, counters and timings.
        fields = " ".join(f"{k}={v}" for k, v in extra.items() if v is not None)
        log.log(
            level,
            "llm request %s provider=%s model=%s request=%s attempt=%d status=%s latency_ms=%d %s",
            logctx.fields() or "call_id=-",
            self._name,
            model,
            request_id,
            attempt,
            status,
            round((time.monotonic() - started) * 1000),
            fields,
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        max_retries: int | None = None,
        timeout_s: float | None = None,
        deadline: float | None = None,
        role: str | None = None,
    ) -> AsyncIterator[str | LLMUsage | LLMServed]:
        """Stream one generation. Retries transient failures only while nothing has been yielded, so a caller
        never hears the same words twice. `deadline` (time.monotonic()) bounds every attempt, backoff and 429
        wait: once it cannot fit another attempt, the last error is raised instead of making the caller wait.
        Yields an LLMServed (which model actually answered) before the first text, then text, then LLMUsage."""
        request_id = uuid.uuid4().hex[:8]
        retries = self._max_retries if max_retries is None else max_retries
        per_attempt_timeout = timeout_s or self._timeout
        for attempt in range(retries + 1):
            produced = False
            started = time.monotonic()
            first_token_ms: int | None = None
            try:
                timeout = per_attempt_timeout
                if deadline is not None:
                    remaining = deadline - started
                    if remaining <= 0.05:
                        raise LLMError(f"{self._name} {model}: response deadline reached before this attempt")
                    timeout = min(timeout, remaining)
                self._check_budget(model)
                async with self._http.stream(
                    "POST",
                    self._url,
                    headers=self._headers,
                    json=self._body(messages, model, stream=True, max_tokens=self._max_tokens),
                    timeout=timeout,
                ) as response:
                    if response.status_code >= 400:
                        raise self._http_error(response, await response.aread(), model)
                    usage: LLMUsage | None = None
                    served = model
                    async for line in response.aiter_lines():
                        # Blank lines separate events; ':' lines are keep-alive comments.
                        if not line.startswith("data: "):
                            continue
                        data = line[len("data: ") :]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError as exc:
                            raise LLMError(f"{self._name} sent a malformed stream event") from exc
                        if "error" in chunk:
                            error = chunk["error"]
                            code = error.get("code") if isinstance(error, dict) else None
                            raise LLMError(
                                f"{self._name} stream error: {self._redact(str(error))[:300]}",
                                status=code if isinstance(code, int) else None,
                            )
                        if isinstance(chunk.get("model"), str) and chunk["model"]:
                            served = chunk["model"]
                        if chunk.get("usage"):
                            usage = _usage(chunk)
                        for choice in chunk.get("choices", []):
                            content = choice.get("delta", {}).get("content")
                            if content:
                                if not produced:
                                    produced = True
                                    first_token_ms = round((time.monotonic() - started) * 1000)
                                    yield LLMServed(provider=self._name, model=served, role=role or "-")
                                yield content
                    if not produced:
                        # e.g. a reasoning model that used its whole budget thinking: silence is never a valid reply.
                        raise LLMError(f"{self._name} returned no text")
                    self._log(
                        request_id=request_id,
                        model=model,
                        attempt=attempt,
                        status="ok",
                        started=started,
                        role=role,
                        served_model=served if served != model else None,
                        ttft_ms=first_token_ms,
                    )
                    yield usage or LLMUsage(0, 0, 0.0)
                    return
            except (asyncio.CancelledError, GeneratorExit):
                # The caller interrupted (barge-in, hang-up): record that this request was abandoned, so a
                # cancelled generation is never invisible in the logs. Tokens already generated may still bill.
                self._log(
                    request_id=request_id,
                    model=model,
                    attempt=attempt,
                    status="cancelled",
                    started=started,
                    role=role,
                    produced=produced,
                )
                raise
            except (httpx.TransportError, LLMError) as exc:
                wait = self._retry_wait(exc, model, attempt)
                # Retry only transient failures, only while nothing has been spoken from this generation, and only
                # if the wait plus another attempt still fits inside the response deadline.
                give_up = (
                    produced
                    or attempt == retries
                    or not is_transient(exc)
                    or (deadline is not None and time.monotonic() + wait + 0.25 >= deadline)
                )
                self._log(
                    request_id=request_id,
                    model=model,
                    attempt=attempt,
                    status=f"error:{_status_label(exc)}",
                    started=started,
                    level=logging.WARNING,
                    role=role,
                    final=give_up,
                    reason=_reason(exc),
                )
                if give_up:
                    raise
                await asyncio.sleep(wait)

    def _retry_wait(self, exc: BaseException, model: str, attempt: int) -> float:
        """Backoff with jitter; after a 429, at least until this model's cooldown (Retry-After) has passed."""
        wait = self._backoff(attempt)
        if is_rate_limited(exc):
            wait = max(wait, self.cooldown_remaining(model))
        return wait

    async def complete(self, messages: list[dict[str, str]], model: str) -> tuple[str, LLMUsage]:
        # No retries: a slow outcome-extraction call has nothing waiting on it, so there is no reason to burn a
        # second request on the same provider instead of just letting the caller treat this one failure as final.
        request_id = uuid.uuid4().hex[:8]
        started = time.monotonic()
        try:
            self._check_budget(model)
            response = await self._http.post(
                self._url,
                headers=self._headers,
                json=self._body(messages, model, stream=False, max_tokens=max(self._max_tokens, MIN_COMPLETE_TOKENS)),
                timeout=COMPLETE_TIMEOUT_SECONDS,
            )
            if response.status_code >= 400:
                raise self._http_error(response, response.content, model)
            payload = response.json()
            self._log(request_id=request_id, model=model, attempt=0, status="ok", started=started)
            return payload["choices"][0]["message"]["content"] or "", _usage(payload)
        except (httpx.TransportError, LLMError) as exc:
            self._log(
                request_id=request_id,
                model=model,
                attempt=0,
                status=f"error:{_status_label(exc)}",
                started=started,
                final=True,
            )
            raise
