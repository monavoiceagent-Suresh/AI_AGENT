"""Per-agent LLM provider/model overrides (the Super Admin's workspace choice, resolved by the dashboard)."""

import pytest

from app.calls import effective_llm_provider, llm_model_for_agent, resolve_llm
from app.models import AgentIn, PersonaIn
from app.providers.failover import FailoverLLM
from tests.fakes import make_settings


def agent_in(**overrides) -> AgentIn:
    base = dict(name="A", language="en", workflow="general", instructions="i", greeting="g")
    return AgentIn(**(base | overrides))


class Services:
    def __init__(self, settings, by_provider):
        self.settings, self.llm, self.llm_by_provider = settings, object(), by_provider


def cfg(**kw):
    return make_settings(openrouter_model="or-model", groq_api_key="g", groq_model="groq-model", **kw)


def test_no_override_uses_the_engine_wide_llm_and_the_primary_model():
    svc = Services(cfg(), {})
    agent = agent_in()
    assert resolve_llm(svc, agent) is svc.llm
    assert llm_model_for_agent(agent, svc.settings) == "or-model"


def test_override_picks_that_providers_own_model_not_the_env_providers():
    settings = cfg()  # env primary is OpenRouter
    assert effective_llm_provider(agent_in(llm_provider="groq"), settings) == "groq"
    assert llm_model_for_agent(agent_in(llm_provider="groq"), settings) == "groq-model"
    assert llm_model_for_agent(agent_in(llm_provider="groq", llm_model="custom"), settings) == "custom"


def test_override_without_a_key_falls_back_to_the_global_provider_and_its_model():
    settings = make_settings(openrouter_model="or-model", groq_api_key=None)  # no Groq key
    agent = agent_in(llm_provider="groq", llm_model="groq-only-model")
    assert effective_llm_provider(agent, settings) == "openrouter"
    assert llm_model_for_agent(agent, settings) == "or-model"


def test_resolve_llm_builds_a_failover_from_the_chosen_pair():
    svc = Services(cfg(), {"openrouter": object(), "groq": object()})
    llm = resolve_llm(svc, agent_in(llm_provider="groq", llm_fallback_provider="openrouter"))
    assert isinstance(llm, FailoverLLM)
    assert llm._primary is svc.llm_by_provider["groq"] and llm._fallback is svc.llm_by_provider["openrouter"]
    assert llm._fallback_model == "or-model"


def test_same_primary_and_fallback_is_rejected():
    with pytest.raises(ValueError, match="llm_fallback_provider must differ"):
        agent_in(llm_provider="groq", llm_fallback_provider="groq")
    with pytest.raises(ValueError, match="llmFallbackProvider must differ"):
        PersonaIn(key="ab", name="A", language="en", instructions="i", greeting="g", version=1,
                  llm_provider="groq", llm_fallback_provider="groq")
