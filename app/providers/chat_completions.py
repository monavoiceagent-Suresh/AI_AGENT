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

from app.interfaces import LLMUsage
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


def _status_label(exc: BaseException) -> str:
    """How an attempt's outcome is logged: the HTTP status if there is one, else the exception's type."""
    status = getattr(exc, "status", None)
    return str(status) if status is not None else type(exc).__name__


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
        # A per-process self-throttle (shared by every call, so it is a real "global" budget for this provider)
        # and a short cooldown set after a 429, honouring Retry-After so a known-limited provider is not retried
        # immediately on the next conversation turn.
        self._limiter = limiter
        self._backoff_base = retry_backoff_base_ms / 1000
        self._backoff_max = retry_backoff_max_ms / 1000
        self._cooldown_until = 0.0

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

    def _http_error(self, response: httpx.Response, body: bytes) -> LLMError:
        if response.status_code == 429:
            # Stop hitting a provider that just told us to slow down; the next attempt (this generation's retry,
            # or the next conversation turn) finds the cooldown still active and fails fast instead of asking again.
            self._cooldown_until = time.monotonic() + self._retry_after_seconds(response)
        detail = self._redact(body[:300].decode("utf-8", "replace"))
        return LLMError(f"{self._name} failed ({response.status_code}): {detail}", status=response.status_code)

    def _check_budget(self) -> None:
        """Raise before making a request this provider's own limits say we shouldn't make yet."""
        if time.monotonic() < self._cooldown_until:
            raise LLMError(f"{self._name} is cooling down after a 429", status=429)
        if self._limiter is not None and not self._limiter.try_acquire():
            raise LLMError(f"{self._name} self-throttled: per-minute request budget reached", status=429)

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter before a retry: attempt 0 waits ~base, attempt 1 ~2x base, capped."""
        delay = min(self._backoff_max, self._backoff_base * (2**attempt))
        return delay + random.uniform(0, self._backoff_base)

    def _log(self, *, request_id: str, model: str, attempt: int, status: str, started: float, **extra: Any) -> None:
        # Never includes messages, response bodies or the API key: only counters and timings.
        fields = " ".join(f"{k}={v}" for k, v in extra.items())
        log.info(
            "llm request provider=%s model=%s request=%s attempt=%d status=%s latency_ms=%d %s",
            self._name,
            model,
            request_id,
            attempt,
            status,
            round((time.monotonic() - started) * 1000),
            fields,
        )

    async def stream(self, messages: list[dict[str, str]], model: str) -> AsyncIterator[str | LLMUsage]:
        request_id = uuid.uuid4().hex[:8]
        for attempt in range(self._max_retries + 1):
            produced = False
            started = time.monotonic()
            try:
                self._check_budget()
                async with self._http.stream(
                    "POST",
                    self._url,
                    headers=self._headers,
                    json=self._body(messages, model, stream=True, max_tokens=self._max_tokens),
                    timeout=self._timeout,
                ) as response:
                    if response.status_code >= 400:
                        raise self._http_error(response, await response.aread())
                    usage: LLMUsage | None = None
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
                            raise LLMError(f"{self._name} stream error: {self._redact(str(chunk['error']))[:300]}")
                        if chunk.get("usage"):
                            usage = _usage(chunk)
                        for choice in chunk.get("choices", []):
                            content = choice.get("delta", {}).get("content")
                            if content:
                                produced = True
                                yield content
                    if not produced:
                        # e.g. a reasoning model that used its whole budget thinking: silence is never a valid reply.
                        raise LLMError(f"{self._name} returned no text")
                    self._log(request_id=request_id, model=model, attempt=attempt, status="ok", started=started)
                    yield usage or LLMUsage(0, 0, 0.0)
                    return
            except (httpx.TransportError, LLMError) as exc:
                # Retry only transient failures, and only while nothing has been spoken from this generation.
                give_up = produced or attempt == self._max_retries or not is_transient(exc) or is_rate_limited(exc)
                self._log(
                    request_id=request_id,
                    model=model,
                    attempt=attempt,
                    status=f"error:{_status_label(exc)}",
                    started=started,
                    final=give_up,
                )
                if give_up:
                    raise
                await asyncio.sleep(self._backoff(attempt))

    async def complete(self, messages: list[dict[str, str]], model: str) -> tuple[str, LLMUsage]:
        # No retries: a slow outcome-extraction call has nothing waiting on it, so there is no reason to burn a
        # second request on the same provider instead of just letting the caller treat this one failure as final.
        request_id = uuid.uuid4().hex[:8]
        started = time.monotonic()
        try:
            self._check_budget()
            response = await self._http.post(
                self._url,
                headers=self._headers,
                json=self._body(messages, model, stream=False, max_tokens=max(self._max_tokens, MIN_COMPLETE_TOKENS)),
                timeout=COMPLETE_TIMEOUT_SECONDS,
            )
            if response.status_code >= 400:
                raise self._http_error(response, response.content)
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
