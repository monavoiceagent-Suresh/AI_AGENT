"""OpenAI-compatible chat completions with streaming. OpenRouter and Groq both speak this format.

Only `delta.content` is spoken. Reasoning models (gpt-oss) stream their thinking in a separate reasoning field
(`delta.reasoning` / `delta.reasoning_content`); it is discarded and never reaches the caller.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.interfaces import LLMUsage

# Replies must start within a few seconds, so a single retry is the most a call can afford.
COMPLETE_TIMEOUT_SECONDS = 30
# Post-call outcome extraction returns a short JSON object; it needs room for reasoning, but no more.
MIN_COMPLETE_TOKENS = 600


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

    def _http_error(self, status: int, body: bytes) -> LLMError:
        detail = self._redact(body[:300].decode("utf-8", "replace"))
        return LLMError(f"{self._name} failed ({status}): {detail}", status=status)

    async def stream(self, messages: list[dict[str, str]], model: str) -> AsyncIterator[str | LLMUsage]:
        for attempt in range(self._max_retries + 1):
            produced = False
            try:
                async with self._http.stream(
                    "POST",
                    self._url,
                    headers=self._headers,
                    json=self._body(messages, model, stream=True, max_tokens=self._max_tokens),
                    timeout=self._timeout,
                ) as response:
                    if response.status_code >= 400:
                        raise self._http_error(response.status_code, await response.aread())
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
                    yield usage or LLMUsage(0, 0, 0.0)
                    return
            except (httpx.TransportError, LLMError) as exc:
                # Retry only transient failures, and only while nothing has been spoken from this generation.
                if produced or attempt == self._max_retries or not is_transient(exc) or is_rate_limited(exc):
                    raise

    async def complete(self, messages: list[dict[str, str]], model: str) -> tuple[str, LLMUsage]:
        response = await self._http.post(
            self._url,
            headers=self._headers,
            json=self._body(messages, model, stream=False, max_tokens=max(self._max_tokens, MIN_COMPLETE_TOKENS)),
            timeout=COMPLETE_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            raise self._http_error(response.status_code, response.content)
        payload = response.json()
        return payload["choices"][0]["message"]["content"] or "", _usage(payload)
