import json
import logging

import httpx
import pytest

from app.interfaces import LLMUsage
from app.providers.chat_completions import ChatCompletionsLLM, LLMError
from app.providers.failover import FailoverLLM
from tests.fakes import make_settings

pytestmark = pytest.mark.anyio

GROQ_KEY = "groq-secret-key"
OPENROUTER_KEY = "openrouter-secret-key"
MESSAGES = [{"role": "user", "content": "What time do you open?"}]


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
    """Mock transport: routes each request by host, records (host, model), and returns scripted responses."""

    def __init__(self, primary: list, fallback: list):
        self._scripts = {"openrouter.ai": iter(primary), "api.groq.com": iter(fallback)}
        self.calls: list[tuple[str, str]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.url.host, json.loads(request.content)["model"]))
        step = next(self._scripts[request.url.host])
        if isinstance(step, Exception):
            raise step
        return step

    def hosts(self) -> list[str]:
        return [host for host, _ in self.calls]


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
    )


def groq(http: httpx.AsyncClient) -> ChatCompletionsLLM:
    return ChatCompletionsLLM(
        http,
        name="Groq",
        api_key=GROQ_KEY,
        base_url="https://api.groq.com/openai/v1",
        max_tokens=1024,
        temperature=0.4,
        timeout_ms=8000,
        max_retries=1,
    )


def build(primary: list, fallback: list, *, retries: int = 1) -> tuple[FailoverLLM, Router]:
    router = Router(primary, fallback)
    http = httpx.AsyncClient(transport=httpx.MockTransport(router))
    llm = FailoverLLM(primary=openrouter(http, retries=retries), fallback=groq(http), fallback_model="fallback-model")
    return llm, router


async def stream_text(llm: FailoverLLM, model: str = "primary-model") -> tuple[str, list]:
    items = [item async for item in llm.stream(MESSAGES, model)]
    return "".join(i for i in items if isinstance(i, str)), items


async def test_primary_success_speaks_only_the_answer_not_the_reasoning():
    llm, router = build([ok(thinking("Let me think. "), delta("We open "), delta("at nine."), USAGE)], [])

    text, items = await stream_text(llm)

    assert text == "We open at nine."
    assert "think" not in text
    assert items[-1] == LLMUsage(prompt_tokens=7, completion_tokens=3, cost_usd=0.0001)
    assert router.calls == [("openrouter.ai", "primary-model")]  # the fallback was never touched


async def test_timeout_before_any_text_falls_back_to_groq():
    llm, router = build([httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow")], [ok(delta("Fallback reply."), USAGE)])

    text, _ = await stream_text(llm)

    assert text == "Fallback reply."
    assert router.calls == [
        ("openrouter.ai", "primary-model"),
        ("openrouter.ai", "primary-model"),
        ("api.groq.com", "fallback-model"),
    ]


async def test_a_reply_that_is_only_reasoning_falls_back_instead_of_going_silent():
    # A reasoning model can spend its whole token budget thinking: finish_reason=length and no spoken text.
    only_thinking = ok(thinking("Hmm, let me think about this for a long time. "), USAGE)
    llm, router = build([only_thinking], [ok(delta("Backup answer."), USAGE)], retries=0)

    text, _ = await stream_text(llm)

    assert text == "Backup answer."
    assert router.calls == [("openrouter.ai", "primary-model"), ("api.groq.com", "fallback-model")]


async def test_connection_error_falls_back():
    llm, router = build([httpx.ConnectError("refused"), httpx.ConnectError("refused")], [ok(delta("Backup."), USAGE)])

    text, _ = await stream_text(llm)

    assert text == "Backup." and router.hosts()[-1] == "api.groq.com"


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_rate_limit_and_server_errors_fall_back(status):
    llm, router = build([fail(status), fail(status)], [ok(delta("Backup."), USAGE)])

    text, _ = await stream_text(llm)

    assert text == "Backup."
    assert router.calls[-1] == ("api.groq.com", "fallback-model")


async def test_rate_limit_is_not_retried_on_the_primary():
    llm, router = build([fail(429)], [ok(delta("Backup."), USAGE)])

    await stream_text(llm)

    assert router.hosts() == ["openrouter.ai", "api.groq.com"]  # one primary attempt, then the fallback


async def test_a_transient_primary_error_is_retried_once_before_falling_back():
    llm, router = build([fail(503), ok(delta("Primary again."), USAGE)], [])

    text, _ = await stream_text(llm)

    assert text == "Primary again."
    assert router.hosts() == ["openrouter.ai", "openrouter.ai"]  # no fallback needed


async def test_client_errors_are_not_failed_over():
    llm, router = build([fail(401, '{"error": "invalid key"}')], [ok(delta("Should not run."), USAGE)])

    with pytest.raises(LLMError) as exc:
        await stream_text(llm)

    assert exc.value.status == 401
    assert router.hosts() == ["openrouter.ai"]


async def test_no_fallback_once_speech_has_started():
    # Text reached the caller, then the stream broke: a second generation would repeat it.
    llm, router = build(
        [ok(delta("We open at "), {"error": {"message": "stream broke"}})], [ok(delta("Repeat."), USAGE)]
    )

    with pytest.raises(LLMError, match="stream error"):
        await stream_text(llm)

    assert router.hosts() == ["openrouter.ai"]


async def test_both_providers_failing_raises_the_fallbacks_error():
    llm, router = build([httpx.ReadTimeout("slow"), httpx.ReadTimeout("slow")], [fail(503), fail(503)])

    with pytest.raises(LLMError) as exc:
        await stream_text(llm)

    assert exc.value.status == 503
    assert router.hosts() == ["openrouter.ai", "openrouter.ai", "api.groq.com", "api.groq.com"]


async def test_no_fallback_configured_sends_every_request_to_openrouter():
    router = Router([ok(delta("Only OpenRouter."), USAGE)], [])
    http = httpx.AsyncClient(transport=httpx.MockTransport(router))
    llm = FailoverLLM(primary=openrouter(http), fallback=None, fallback_model="fallback-model")

    text, _ = await stream_text(llm, model="agent-override")

    assert text == "Only OpenRouter."
    assert router.calls == [("openrouter.ai", "agent-override")]


async def test_without_a_fallback_the_primarys_error_is_raised():
    router = Router([fail(503)], [])
    http = httpx.AsyncClient(transport=httpx.MockTransport(router))
    llm = FailoverLLM(primary=openrouter(http), fallback=None, fallback_model="fallback-model")

    with pytest.raises(LLMError) as exc:
        await stream_text(llm)

    assert exc.value.status == 503
    assert router.hosts() == ["openrouter.ai"]


async def test_complete_falls_back_on_server_error_and_not_on_client_error():
    llm, router = build(
        [fail(500)], [httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}], "usage": {}})]
    )
    text, usage = await llm.complete(MESSAGES, "primary-model")
    assert text == "{}" and router.calls[-1] == ("api.groq.com", "fallback-model")

    llm, router = build([fail(400)], [])
    with pytest.raises(LLMError):
        await llm.complete(MESSAGES, "primary-model")
    assert router.hosts() == ["openrouter.ai"]


async def test_api_keys_never_appear_in_errors_or_logs(caplog):
    # A provider that echoes the credential in its error body must not leak it.
    echo = f'{{"error": "bad key {OPENROUTER_KEY}"}}'
    llm, _ = build([fail(401, echo)], [])
    caplog.set_level(logging.DEBUG)

    with pytest.raises(LLMError) as exc:
        await stream_text(llm)

    assert OPENROUTER_KEY not in str(exc.value) and "***" in str(exc.value)
    assert all(OPENROUTER_KEY not in r.getMessage() for r in caplog.records)


async def test_failover_warning_names_the_failure_without_its_message(caplog):
    llm, _ = build([fail(503, "upstream overloaded for account 42")] * 2, [ok(delta("Backup."), USAGE)])
    caplog.set_level(logging.WARNING)

    await stream_text(llm)

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings and "LLMError 503" in warnings[0] and "account 42" not in warnings[0]


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


async def test_openrouter_is_primary_and_a_timeout_falls_back_to_groqs_endpoint_and_model():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers["authorization"], json.loads(request.content)["model"]))
        if request.url.host == "openrouter.ai":
            raise httpx.ReadTimeout("no first token in time")
        return ok(delta("Backup."), USAGE)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = make_settings(openrouter_model="or-model", openrouter_max_retries=0, groq_api_key="groq-key")
    primary = ChatCompletionsLLM(
        http,
        name="OpenRouter",
        api_key=settings.openrouter_api_key.get_secret_value(),
        base_url=settings.openrouter_base_url,
        max_tokens=settings.openrouter_max_output_tokens,
        temperature=settings.openrouter_temperature,
        timeout_ms=settings.openrouter_timeout_ms,
        max_retries=settings.openrouter_max_retries,
    )
    fallback = ChatCompletionsLLM(
        http,
        name="Groq",
        api_key=settings.groq_api_key.get_secret_value(),
        base_url=settings.groq_base_url,
        max_tokens=settings.groq_max_output_tokens,
        temperature=settings.groq_temperature,
        timeout_ms=settings.groq_timeout_ms,
        max_retries=settings.groq_max_retries,
    )
    llm = FailoverLLM(primary=primary, fallback=fallback, fallback_model=settings.groq_model)

    text, _ = await stream_text(llm, settings.primary_llm_model)

    assert text == "Backup."
    assert seen == [
        ("https://openrouter.ai/api/v1/chat/completions", "Bearer o", "or-model"),
        ("https://api.groq.com/openai/v1/chat/completions", "Bearer groq-key", "openai/gpt-oss-20b"),
    ]  # one attempt on each provider, and Groq gets its own model, not OpenRouter's


def test_the_primary_model_is_openrouters_and_the_old_setting_name_still_works():
    # make_settings sets it through openrouter_fallback_model, the name it had when OpenRouter was the fallback.
    assert make_settings().primary_llm_model == "fallback-model"
    assert make_settings(openrouter_model="or-model").primary_llm_model == "or-model"
    assert make_settings(groq_api_key="g").primary_llm_model == make_settings(groq_api_key=None).primary_llm_model
    assert make_settings(groq_api_key=None).groq_api_key is None  # Groq is an optional fallback
