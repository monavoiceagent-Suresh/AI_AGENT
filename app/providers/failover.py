"""Primary LLM with an automatic fallback to a second provider.

The fallback is taken only when the primary fails before it has produced any text: timeouts, connection errors,
rate limits and server errors. If the primary fails after speaking has started, the error is raised instead, because
a second attempt would make the caller hear the same words again. Client errors (a bad key, an unknown model) are not
failed over, so a misconfiguration stays visible instead of being silently routed elsewhere.
"""

import logging
from collections.abc import AsyncIterator

from app.interfaces import LLM, LLMUsage
from app.providers.chat_completions import is_transient

log = logging.getLogger(__name__)


def _describe(exc: BaseException) -> str:
    """Name the failure without its message: provider bodies can echo prompts or credentials."""
    status = getattr(exc, "status", None)
    return type(exc).__name__ + (f" {status}" if status else "")


class FailoverLLM:
    def __init__(self, *, primary: LLM, fallback: LLM | None, fallback_model: str):
        """`fallback` is None when no fallback provider is configured; then the primary's errors are raised."""
        self._primary = primary
        self._fallback = fallback
        self._fallback_model = fallback_model

    async def stream(self, messages: list[dict[str, str]], model: str) -> AsyncIterator[str | LLMUsage]:
        if self._fallback is None:
            async for item in self._primary.stream(messages, model):
                yield item
            return
        produced = False
        try:
            async for item in self._primary.stream(messages, model):
                if isinstance(item, str):
                    produced = True
                yield item
        except Exception as exc:
            if produced or not is_transient(exc):
                raise
            log.warning("primary LLM failed before replying (%s); using fallback", _describe(exc))
            async for item in self._fallback.stream(messages, self._fallback_model):
                yield item

    async def complete(self, messages: list[dict[str, str]], model: str) -> tuple[str, LLMUsage]:
        if self._fallback is None:
            return await self._primary.complete(messages, model)
        try:
            return await self._primary.complete(messages, model)
        except Exception as exc:
            if not is_transient(exc):
                raise
            log.warning("primary LLM failed (%s); using fallback", _describe(exc))
            return await self._fallback.complete(messages, self._fallback_model)
