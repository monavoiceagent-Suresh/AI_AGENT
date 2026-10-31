"""Caps how many LLM requests one phone call may make, on top of each provider's own per-minute throttle.

This is the backstop against a single call making an unbounded number of requests: a stuck conversation loop,
or a run of failures each triggering another attempt. The count is reset for every new call and is never shared
between calls, unlike the provider-level TokenBucket, which is shared by the whole process.
"""

import logging
from collections.abc import AsyncIterator

from app.interfaces import LLM, LLMUsage
from app.providers.chat_completions import LLMError

log = logging.getLogger(__name__)


class CallBudgetExceeded(LLMError):
    """This call already made as many LLM requests as LLM_MAX_REQUESTS_PER_CALL allows."""


class BudgetedLLM:
    """Wraps the shared LLM for the lifetime of one call. `max_requests <= 0` means unlimited."""

    def __init__(self, inner: LLM, *, max_requests: int, call_sid: str = "") -> None:
        self._inner = inner
        self._max = max_requests
        self._call_sid = call_sid
        self.request_count = 0

    def _reserve(self) -> None:
        if self._max > 0 and self.request_count >= self._max:
            log.warning(
                "call %s reached its LLM request budget (%d); no further requests will be made",
                self._call_sid or "?",
                self._max,
            )
            raise CallBudgetExceeded(f"call {self._call_sid or '?'} reached its LLM request budget ({self._max})")
        self.request_count += 1

    async def stream(self, messages: list[dict[str, str]], model: str) -> AsyncIterator[str | LLMUsage]:
        self._reserve()
        async for item in self._inner.stream(messages, model):
            yield item

    async def complete(self, messages: list[dict[str, str]], model: str) -> tuple[str, LLMUsage]:
        self._reserve()
        return await self._inner.complete(messages, model)
