"""Guards against excessive LLM requests: per-provider self-throttling, 429 cooldown with backoff+jitter,
and the per-call request budget. See tests/test_llm.py for the provider/fallback selection itself, and
test_session.py for the orchestrator-level duplicate-utterance and repeated-failure guards.
"""

import json

import httpx
import pytest

from app.providers.budget import BudgetedLLM, CallBudgetExceeded
from app.providers.chat_completions import ChatCompletionsLLM, LLMError
from app.providers.rate_limit import TokenBucket

pytestmark = pytest.mark.anyio

MESSAGES = [{"role": "user", "content": "What time do you open?"}]


def sse(*events: dict) -> bytes:
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return body.encode()


def delta(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}}]}


USAGE = {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3, "cost": 0.0001}}


def ok() -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse(delta("Hi."), USAGE))


def rate_limited(retry_after: str | None = None) -> httpx.Response:
    headers = {"retry-after": retry_after} if retry_after else {}
    return httpx.Response(429, headers=headers, content=b"{}")


class CountingTransport:
    """Returns each scripted response in order; raises if asked for more than were scripted."""

    def __init__(self, *responses: httpx.Response) -> None:
        self._responses = list(responses)
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if not self._responses:
            raise AssertionError("the provider was called more times than this test expected")
        return self._responses.pop(0)


def client(transport: CountingTransport, *, limiter: TokenBucket | None = None, retries: int = 0) -> ChatCompletionsLLM:
    http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    return ChatCompletionsLLM(
        http,
        name="OpenRouter",
        api_key="k",
        base_url="https://openrouter.ai/api/v1",
        max_tokens=250,
        temperature=0.4,
        timeout_ms=4000,
        max_retries=retries,
        limiter=limiter,
    )


async def drain(llm: ChatCompletionsLLM) -> None:
    async for _ in llm.stream(MESSAGES, "m"):
        pass


# ---- per-provider self-throttle (the per-minute budget) --------------------------------------------------


async def test_the_nth_plus_one_request_is_rejected_without_calling_the_provider():
    transport = CountingTransport(ok(), ok())  # only two genuine responses are ever expected
    llm = client(transport, limiter=TokenBucket(2))

    await drain(llm)
    await drain(llm)
    with pytest.raises(LLMError) as exc:
        await drain(llm)

    assert exc.value.status == 429
    assert transport.calls == 2  # the third request never reached the provider


async def test_a_zero_or_negative_limit_means_unlimited():
    transport = CountingTransport(ok(), ok(), ok())
    llm = client(transport, limiter=TokenBucket(0))

    for _ in range(3):
        await drain(llm)

    assert transport.calls == 3


async def test_the_limiter_refills_after_its_window(monkeypatch):
    now = [0.0]
    bucket = TokenBucket(1, clock=lambda: now[0])
    assert bucket.try_acquire()
    assert not bucket.try_acquire()
    now[0] = 60.0
    assert bucket.try_acquire()


# ---- 429 cooldown, honouring Retry-After ------------------------------------------------------------------


async def test_a_429_with_retry_after_blocks_the_next_request_without_calling_the_provider():
    # Only one response is scripted: if the cooldown did not hold, the second call would raise on the
    # transport running out of scripts instead of failing fast locally.
    transport = CountingTransport(rate_limited(retry_after="30"))
    llm = client(transport)

    with pytest.raises(LLMError) as exc:
        await drain(llm)
    assert exc.value.status == 429
    assert transport.calls == 1

    with pytest.raises(LLMError) as exc2:
        await drain(llm)
    assert exc2.value.status == 429
    assert transport.calls == 1  # still 1: the second attempt never reached the provider


async def test_a_429_with_no_retry_after_still_cools_down():
    transport = CountingTransport(rate_limited())
    llm = client(transport)

    with pytest.raises(LLMError):
        await drain(llm)
    with pytest.raises(LLMError):
        await drain(llm)

    assert transport.calls == 1


async def test_a_plain_server_error_does_not_trigger_the_429_cooldown():
    transport = CountingTransport(httpx.Response(500, content=b"{}"), ok())
    llm = client(transport)

    with pytest.raises(LLMError):
        await drain(llm)
    await drain(llm)  # not blocked: a 500 is not a rate limit

    assert transport.calls == 2


# ---- exponential backoff with jitter on retry --------------------------------------------------------------


async def test_a_retry_waits_with_bounded_backoff_before_trying_again(monkeypatch):
    transport = CountingTransport(httpx.Response(503, content=b"{}"), ok())
    llm = client(transport, retries=1)
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("app.providers.chat_completions.asyncio.sleep", fake_sleep)

    await drain(llm)

    assert transport.calls == 2
    assert len(sleeps) == 1
    # attempt 0: base (0.3s default) up to base + its own jitter (another 0.3s), never the max_ms cap (2s).
    assert 0.3 <= sleeps[0] <= 0.6


async def test_backoff_never_exceeds_the_configured_cap(monkeypatch):
    transport = CountingTransport(*(httpx.Response(503, content=b"{}") for _ in range(4)), ok())
    http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    llm = ChatCompletionsLLM(
        http,
        name="OpenRouter",
        api_key="k",
        base_url="https://openrouter.ai/api/v1",
        max_tokens=250,
        temperature=0.4,
        timeout_ms=4000,
        max_retries=4,
        retry_backoff_base_ms=300,
        retry_backoff_max_ms=500,
    )
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("app.providers.chat_completions.asyncio.sleep", fake_sleep)

    await drain(llm)

    assert all(s <= 0.5 + 0.3 for s in sleeps)  # capped delay plus its own jitter, never unbounded growth


async def test_a_429_is_never_retried_inline_even_when_retries_are_allowed():
    transport = CountingTransport(rate_limited())
    llm = client(transport, retries=3)

    with pytest.raises(LLMError) as exc:
        await drain(llm)

    assert exc.value.status == 429
    assert transport.calls == 1  # no retry burned on a rate limit; the caller should use the fallback instead


# ---- per-call request budget --------------------------------------------------------------------------------


class _CountingInner:
    def __init__(self) -> None:
        self.stream_calls = 0
        self.complete_calls = 0

    async def stream(self, messages, model):
        self.stream_calls += 1
        yield "ok "

    async def complete(self, messages, model):
        self.complete_calls += 1
        return "ok", None


async def test_the_call_budget_stops_delegating_once_it_is_used_up():
    inner = _CountingInner()
    llm = BudgetedLLM(inner, max_requests=2, call_sid="CAtest")

    async for _ in llm.stream(MESSAGES, "m"):
        pass
    await llm.complete(MESSAGES, "m")
    with pytest.raises(CallBudgetExceeded):
        async for _ in llm.stream(MESSAGES, "m"):
            pass

    assert inner.stream_calls == 1 and inner.complete_calls == 1  # the third request never reached the inner LLM


async def test_a_non_positive_call_budget_means_unlimited():
    inner = _CountingInner()
    llm = BudgetedLLM(inner, max_requests=0)

    for _ in range(5):
        async for _ in llm.stream(MESSAGES, "m"):
            pass

    assert inner.stream_calls == 5


async def test_each_call_gets_its_own_budget_not_a_shared_one():
    inner = _CountingInner()
    first, second = BudgetedLLM(inner, max_requests=1), BudgetedLLM(inner, max_requests=1)

    async for _ in first.stream(MESSAGES, "m"):
        pass
    async for _ in second.stream(MESSAGES, "m"):  # a fresh budget for a different call, not exhausted by the first
        pass

    assert inner.stream_calls == 2
