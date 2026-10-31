import json
import logging

import httpx
import pytest

from app.interfaces import LLMUsage
from app.main import build_llm
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


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def configured(provider: str, handler, **overrides):
    """The LLM that build_llm makes for LLM_PROVIDER=`provider`, with both keys set unless overridden."""
    settings = make_settings(llm_provider=provider, openrouter_model="or-model", groq_api_key="groq-key", **overrides)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return build_llm(http, settings), settings


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        (
            "openrouter",
            [(OPENROUTER_URL, "Bearer o", "or-model"), (GROQ_URL, "Bearer groq-key", "openai/gpt-oss-20b")],
        ),
        (
            "groq",
            [(GROQ_URL, "Bearer groq-key", "openai/gpt-oss-20b"), (OPENROUTER_URL, "Bearer o", "or-model")],
        ),
    ],
)
async def test_llm_provider_decides_which_one_answers_first_and_the_other_is_the_fallback(provider, expected):
    seen = []
    first_host = httpx.URL(expected[0][0]).host

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers["authorization"], json.loads(request.content)["model"]))
        if request.url.host == first_host:
            raise httpx.ReadTimeout("no first token in time")
        return ok(delta("Backup."), USAGE)

    llm, settings = configured(provider, handler)

    text, _ = await stream_text(llm, settings.primary_llm_model)

    assert text == "Backup."
    # One attempt on each provider, and each provider is asked for its own model, never the other's.
    assert seen == expected


@pytest.mark.parametrize("value", ["GROQ", "Groq", " groq "])
def test_groq_can_be_written_in_any_case(value):
    settings = make_settings(llm_provider=value, groq_api_key="g")
    assert settings.llm_provider == "groq" and settings.primary_llm_model == "openai/gpt-oss-20b"


@pytest.mark.parametrize("value", ["OPENROUTER", "OpenRouter", " openrouter "])
def test_openrouter_can_be_written_in_any_case(value):
    settings = make_settings(llm_provider=value, openrouter_model="or-model")
    assert settings.llm_provider == "openrouter" and settings.primary_llm_model == "or-model"


def test_a_missing_key_for_the_chosen_primary_is_a_startup_error():
    with pytest.raises(ValueError, match="LLM_PROVIDER=groq requires GROQ_API_KEY"):
        make_settings(llm_provider="groq", groq_api_key=None)
    with pytest.raises(ValueError, match="LLM_PROVIDER=openrouter requires OPENROUTER_API_KEY"):
        make_settings(llm_provider="openrouter", openrouter_api_key=None)
    with pytest.raises(ValueError, match="OPENROUTER_MODEL"):
        make_settings(llm_provider="groq", groq_api_key="g", openrouter_model="")
    with pytest.raises(ValueError):
        make_settings(llm_provider="gemini")


@pytest.mark.parametrize(
    ("provider", "other_key", "only_url"),
    [("groq", {"openrouter_api_key": None}, GROQ_URL), ("openrouter", {"groq_api_key": None}, OPENROUTER_URL)],
)
async def test_the_other_provider_is_optional_without_it_there_is_no_fallback(provider, other_key, only_url):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return fail(503)

    settings = make_settings(llm_provider=provider, openrouter_model="or-model", **{"groq_api_key": "g", **other_key})
    llm = build_llm(httpx.AsyncClient(transport=httpx.MockTransport(handler)), settings)

    with pytest.raises(LLMError) as exc:
        await stream_text(llm)

    assert exc.value.status == 503
    assert seen == [only_url]  # the primary's own error, and nothing else was tried


@pytest.mark.parametrize(("provider", "primary_name"), [("openrouter", "OpenRouter"), ("groq", "Groq")])
def test_timeouts_default_by_role_and_an_explicit_setting_wins(provider, primary_name):
    llm, _ = configured(provider, lambda request: fail(500))

    assert llm._primary._name == primary_name
    assert (llm._primary._timeout, llm._primary._max_retries) == (4.0, 0)  # gives up quickly
    assert (llm._fallback._timeout, llm._fallback._max_retries) == (8.0, 1)  # nothing is behind it

    llm, _ = configured(provider, lambda request: fail(500), groq_timeout_ms=1234, groq_max_retries=0)
    groq_client = llm._primary if provider == "groq" else llm._fallback
    assert (groq_client._timeout, groq_client._max_retries) == (1.234, 0)


def test_the_old_openrouter_fallback_model_name_still_sets_the_model():
    # make_settings uses openrouter_fallback_model, the name this setting had when OpenRouter was always the fallback.
    assert make_settings().primary_llm_model == "fallback-model"
    assert make_settings(openrouter_model="or-model").primary_llm_model == "or-model"
