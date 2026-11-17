"""OpenRouter-only LLM: the primary model, then a different fallback model on the same gateway, inside one response
deadline. Covers success, 429/5xx/timeout/connection failures, both models failing, cancellation, concurrency under
a rate limit, which model served each reply, and that no other gateway is ever called."""

import asyncio
import json
import logging

import httpx
import pytest

from app.interfaces import LLMServed, LLMUsage
from app.main import build_llm, failover_for
from app.providers.chat_completions import ChatCompletionsLLM, LLMError
from app.providers.failover import FailoverLLM, LLMUnavailable
from tests.fakes import make_settings

pytestmark = pytest.mark.anyio

OPENROUTER_KEY = "openrouter-secret-key"
MESSAGES = [{"role": "user", "content": "What time do you open?"}]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def sse(*events: dict) -> bytes:
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"
    return body.encode()


def delta(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}}]}


def thinking(text: str) -> dict:
    return {"choices": [{"delta": {"reasoning": text}}]}


USAGE = {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3, "cost": 0.0001}}


def ok(*events: dict) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse(*events))


def fail(status: int, body: str = "{}") -> httpx.Response:
    return httpx.Response(status, content=body.encode())


class Router:
    """Mock OpenRouter: answers each request from the script of the model it asked for, and records the model."""

    def __init__(self, **scripts: list):
        self._scripts = {model.replace("_", "-"): iter(steps) for model, steps in scripts.items()}
        self.models: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.host == "openrouter.ai"  # OpenRouter is the only gateway: nothing else is ever called
        model = json.loads(request.content)["model"]
        self.models.append(model)
        step = next(self._scripts[model])
        if isinstance(step, Exception):
            raise step
        return step


def openrouter(http: httpx.AsyncClient, *, retries: int = 0) -> ChatCompletionsLLM:
    return ChatCompletionsLLM(
        http,
        name="OpenRouter",
        api_key=OPENROUTER_KEY,
        base_url="https://openrouter.ai/api/v1",
        max_tokens=250,
        temperature=0.4,
        timeout_ms=4000,
        max_retries=retries,
        retry_backoff_base_ms=1,
        retry_backoff_max_ms=2,
    )


def build(primary: list, fallback: list, *, fallback_retries: int = 1, deadline_s: float = 8.0):
    """The production shape: one OpenRouter client, primary model then fallback model."""
    router = Router(primary_model=primary, fallback_model=fallback)
    client = openrouter(httpx.AsyncClient(transport=httpx.MockTransport(router)))
    llm = FailoverLLM(
        primary=client,
        fallback=client,
        fallback_model="fallback-model",
        primary_retries=0,
        fallback_retries=fallback_retries,
        deadline_s=deadline_s,
    )
    return llm, router, client


async def stream_text(llm, model: str = "primary-model") -> tuple[str, list]:
    items = [item async for item in llm.stream(MESSAGES, model)]
    return "".join(i for i in items if isinstance(i, str)), items


def served(items: list) -> LLMServed:
    return next(i for i in items if isinstance(i, LLMServed))


async def test_primary_success_speaks_only_the_answer_and_names_the_model():
    llm, router, _ = build([ok(thinking("Let me think. "), delta("We open "), delta("at nine."), USAGE)], [])

    text, items = await stream_text(llm)

    assert text == "We open at nine."
    assert items[-1] == LLMUsage(prompt_tokens=7, completion_tokens=3, cost_usd=0.0001)
    assert served(items) == LLMServed(provider="OpenRouter", model="primary-model", role="primary")
    assert router.models == ["primary-model"]  # the fallback model was never asked


async def test_the_served_model_reported_by_openrouter_is_kept():
    event = {**delta("Hi."), "model": "google/gemini-2.5-flash-lite-001"}
    llm, _, _ = build([ok(event, USAGE)], [])
    _, items = await stream_text(llm)
    assert served(items).model == "google/gemini-2.5-flash-lite-001"


@pytest.mark.parametrize("status", [429, 500, 502, 503])
async def test_rate_limit_and_server_errors_go_straight_to_the_fallback_model(status):
    llm, router, _ = build([fail(status)], [ok(delta("Backup."), USAGE)])

    text, items = await stream_text(llm)

    assert text == "Backup."
    assert router.models == ["primary-model", "fallback-model"]  # one primary attempt, no inline retry
    assert served(items) == LLMServed(provider="OpenRouter", model="fallback-model", role="fallback")


async def test_a_timeout_before_any_text_falls_back():
    llm, router, _ = build([httpx.ReadTimeout("slow")], [ok(delta("Fallback reply."), USAGE)])
    text, _ = await stream_text(llm)
    assert text == "Fallback reply." and router.models == ["primary-model", "fallback-model"]


async def test_a_connection_error_falls_back():
    llm, router, _ = build([httpx.ConnectError("refused")], [ok(delta("Backup."), USAGE)])
    text, _ = await stream_text(llm)
    assert text == "Backup." and router.models[-1] == "fallback-model"


async def test_a_reply_that_is_only_reasoning_falls_back_instead_of_going_silent():
    only_thinking = ok(thinking("Hmm, let me think about this for a long time. "), USAGE)
    llm, router, _ = build([only_thinking], [ok(delta("Backup answer."), USAGE)])
    text, _ = await stream_text(llm)
    assert text == "Backup answer." and router.models == ["primary-model", "fallback-model"]


async def test_a_model_specific_client_error_falls_back_but_an_account_error_does_not():
    # An unknown/unavailable primary model: another model on the same key can still answer.
    llm, router, _ = build([fail(404, '{"error": "no such model"}')], [ok(delta("Backup."), USAGE)])
    assert (await stream_text(llm))[0] == "Backup."

    # A bad key or no credits fails identically for every model: switching models cannot help.
    for status in (401, 402, 403):
        llm, router, _ = build([fail(status)], [ok(delta("Should not run."), USAGE)])
        with pytest.raises(LLMError) as exc:
            await stream_text(llm)
        assert exc.value.status == status and router.models == ["primary-model"]


async def test_both_models_failing_is_bounded_and_raises_a_recordable_reason():
    llm, router, _ = build([fail(503)], [fail(503), fail(502)])

    with pytest.raises(LLMUnavailable) as exc:
        await stream_text(llm)

    assert router.models == ["primary-model", "fallback-model", "fallback-model"]  # 3 requests at most
    assert exc.value.reason == "primary=upstream_error fallback=upstream_error"


async def test_a_transient_fallback_error_recovers_on_its_one_retry():
    llm, router, _ = build([fail(429)], [fail(503), ok(delta("Second try."), USAGE)])
    text, _ = await stream_text(llm)
    assert text == "Second try." and router.models == ["primary-model", "fallback-model", "fallback-model"]


async def test_no_fallback_once_speech_has_started():
    # Text reached the caller, then the stream broke: a second generation would repeat it.
    llm, router, _ = build([ok(delta("We open at "), {"error": {"message": "stream broke"}})], [ok(delta("Repeat."))])

    with pytest.raises(LLMError, match="stream error"):
        await stream_text(llm)

    assert router.models == ["primary-model"]


async def test_the_response_deadline_stops_retries_that_would_overrun_it():
    # The fallback is rate limited for 30 s: waiting for that cannot fit the deadline, so no retry is made and
    # the caller gets the controlled failure path at once instead of silence.
    llm, router, _ = build(
        [fail(503)], [httpx.Response(429, headers={"retry-after": "30"}, content=b"{}"), ok(delta("late"))]
    )

    with pytest.raises(LLMUnavailable):
        await asyncio.wait_for(stream_text(llm), 2)

    assert router.models == ["primary-model", "fallback-model"]


async def test_without_a_fallback_model_the_primarys_error_is_raised():
    router = Router(primary_model=[fail(503)])
    client = openrouter(httpx.AsyncClient(transport=httpx.MockTransport(router)))
    llm = FailoverLLM(primary=client, fallback=client, fallback_model="")

    with pytest.raises(LLMError) as exc:
        await stream_text(llm)

    assert exc.value.status == 503 and router.models == ["primary-model"]


async def test_a_429_cools_down_only_that_model_so_concurrent_calls_use_the_fallback_at_once():
    """One call's 429 must not break unrelated concurrent calls: while the primary model cools down, every other
    call skips it locally (no request) and is answered by the fallback model on its first attempt."""
    router = Router(
        primary_model=[httpx.Response(429, headers={"retry-after": "30"}, content=b"{}")],
        fallback_model=[ok(delta(f"Answer {i}."), USAGE) for i in range(4)],
    )
    client = openrouter(httpx.AsyncClient(transport=httpx.MockTransport(router)))

    def call_llm():
        return FailoverLLM(primary=client, fallback=client, fallback_model="fallback-model", primary_retries=0)

    first, _ = await stream_text(call_llm())
    others = await asyncio.gather(*(stream_text(call_llm()) for _ in range(3)))

    assert first == "Answer 0." and sorted(text for text, _ in others) == ["Answer 1.", "Answer 2.", "Answer 3."]
    assert router.models.count("primary-model") == 1  # the cooling-down model was not asked again
    assert client.cooldown_remaining("primary-model") > 0 and client.cooldown_remaining("fallback-model") == 0


async def test_a_cancelled_generation_is_logged_and_never_failed_over(caplog):
    gate = asyncio.Event()

    async def slow_body():
        yield b'data: {"choices": [{"delta": {"content": "Partial "}}]}\n\n'
        await gate.wait()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=slow_body())

    client = openrouter(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    llm = FailoverLLM(primary=client, fallback=client, fallback_model="fallback-model", primary_retries=0)
    caplog.set_level(logging.INFO)
    received: list = []

    async def consume():
        async for item in llm.stream(MESSAGES, "primary-model"):
            received.append(item)

    task = asyncio.create_task(consume())
    async with asyncio.timeout(2):
        while not any(isinstance(i, str) for i in received):
            await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    lines = [r.getMessage() for r in caplog.records if "llm request" in r.getMessage()]
    assert any("status=cancelled" in line and "produced=True" in line for line in lines)
    assert not any("fallback-model" in line for line in lines)  # a cancel is never retried: no duplicate generation


async def test_complete_falls_back_on_a_model_failure_and_not_on_an_account_error():
    llm, router, _ = build(
        [fail(500)], [httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}], "usage": {}})]
    )
    text, _ = await llm.complete(MESSAGES, "primary-model")
    assert text == "{}" and router.models == ["primary-model", "fallback-model"]

    llm, router, _ = build([fail(401)], [])
    with pytest.raises(LLMError):
        await llm.complete(MESSAGES, "primary-model")
    assert router.models == ["primary-model"]


async def test_api_keys_never_appear_in_errors_or_logs(caplog):
    # A provider that echoes the credential in its error body must not leak it.
    echo = f'{{"error": "bad key {OPENROUTER_KEY}"}}'
    llm, _, _ = build([fail(401, echo)], [])
    caplog.set_level(logging.DEBUG)

    with pytest.raises(LLMError) as exc:
        await stream_text(llm)

    assert OPENROUTER_KEY not in str(exc.value) and "***" in str(exc.value)
    assert all(OPENROUTER_KEY not in r.getMessage() for r in caplog.records)


async def test_failover_logs_name_the_failure_without_the_provider_message(caplog):
    llm, _, _ = build([fail(503, "upstream overloaded for account 42")], [ok(delta("Backup."), USAGE)])
    caplog.set_level(logging.WARNING)

    await stream_text(llm)

    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("LLMError 503" in m and "fallback-model" in m for m in messages)
    assert any("llm fallback served" in m and "primary=upstream_error" in m for m in messages)
    assert all("account 42" not in m for m in messages)


async def test_openrouter_request_asks_for_a_stream_with_the_token_budget():
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return ok(delta("Hi."), USAGE)

    primary = openrouter(httpx.AsyncClient(transport=httpx.MockTransport(handler)), retries=1)

    [item async for item in primary.stream(MESSAGES, "primary-model")]

    assert (
        captured[0]["stream"] is True and captured[0]["max_tokens"] == 250 and captured[0]["model"] == "primary-model"
    )


async def test_the_engine_default_is_openrouter_primary_then_its_fallback_model():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        seen.append((str(request.url), request.headers["authorization"], model))
        if model == "or-model":
            return fail(429)
        return ok(delta("Backup."), USAGE)

    settings = make_settings(openrouter_model="or-model", llm_fallback_model="or-fallback", groq_api_key="g")
    llm = build_llm(httpx.AsyncClient(transport=httpx.MockTransport(handler)), settings)

    text, _ = await stream_text(llm, "or-model")

    assert text == "Backup."
    # Both on OpenRouter with the OpenRouter key; a Groq key, even if set, is never used.
    assert seen == [(OPENROUTER_URL, "Bearer o", "or-model"), (OPENROUTER_URL, "Bearer o", "or-fallback")]


def test_groq_is_not_accepted_as_the_llm_gateway():
    with pytest.raises(ValueError, match="OpenRouter is the only supported LLM gateway"):
        make_settings(llm_provider="groq", groq_api_key="g")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is required"):
        make_settings(openrouter_api_key=None)
    with pytest.raises(ValueError):
        make_settings(llm_provider="gemini")


@pytest.mark.parametrize("value", ["OPENROUTER", "OpenRouter", " openrouter "])
def test_openrouter_can_be_written_in_any_case(value):
    settings = make_settings(llm_provider=value, openrouter_model="or-model")
    assert settings.llm_provider == "openrouter" and settings.primary_llm_model == "or-model"


def test_fallback_model_defaults_and_can_be_disabled():
    assert make_settings().llm_fallback_model == "openai/gpt-4o-mini"
    assert make_settings(llm_fallback_model="").llm_fallback_model is None


def test_a_fallback_equal_to_the_primary_means_no_fallback():
    settings = make_settings(openrouter_model="same", llm_fallback_model="same")
    llm = build_llm(httpx.AsyncClient(transport=httpx.MockTransport(lambda r: fail(500))), settings)
    assert llm.fallback_model is None


def test_timeouts_default_by_role_and_an_explicit_setting_wins():
    client = openrouter(httpx.AsyncClient())
    llm = failover_for(client, make_settings(), "fb")
    assert llm._primary_opts["timeout_s"] == 4.0 and llm._primary_opts["max_retries"] == 0  # gives up quickly
    assert llm._fallback_opts["timeout_s"] == 8.0 and llm._fallback_opts["max_retries"] == 1  # nothing behind it
    assert llm._deadline_s == 8.0

    llm = failover_for(client, make_settings(openrouter_timeout_ms=1234, openrouter_max_retries=2), "fb")
    assert llm._primary_opts["timeout_s"] == 1.234 and llm._fallback_opts["max_retries"] == 2
    assert llm._primary_opts["max_retries"] == 0  # with a fallback model, the fallback is the primary's retry


def test_production_timeout_and_retry_settings_cannot_starve_the_fallback():
    # Production sets OPENROUTER_TIMEOUT_MS=8000 and OPENROUTER_MAX_RETRIES=1 (from before the fallback model
    # existed). Applied to the primary as-is, a hung primary would use the whole 8 s deadline (or retry into it)
    # and the fallback model would never get a turn.
    client = openrouter(httpx.AsyncClient())
    prod = make_settings(openrouter_timeout_ms=8000, openrouter_max_retries=1)
    llm = failover_for(client, prod, "openai/gpt-4o-mini")
    assert llm._primary_opts["timeout_s"] == 4.0  # at most half the 8 s deadline
    assert llm._primary_opts["max_retries"] == 0
    assert llm._fallback_opts["timeout_s"] == 8.0 and llm._fallback_opts["max_retries"] == 1

    # Without a fallback model the primary keeps its configured retries (nothing else could answer).
    alone = failover_for(client, prod, None)
    assert alone._primary_opts["max_retries"] == 1 and alone._primary_opts["timeout_s"] == 8.0


def test_the_old_openrouter_fallback_model_name_still_sets_the_primary_model():
    # make_settings uses openrouter_fallback_model, an old alias of OPENROUTER_MODEL (not LLM_FALLBACK_MODEL).
    assert make_settings().primary_llm_model == "fallback-model"
    assert make_settings(openrouter_model="or-model").primary_llm_model == "or-model"
