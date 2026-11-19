"""Per-workspace LLM model choice (the Super Admin's setting, resolved by the dashboard): OpenRouter only, a primary
model and a fallback model, each falling back to the engine's own default when not set."""

import pytest

from app.calls import effective_llm_provider, llm_fallback_model_for_agent, llm_model_for_agent, resolve_llm
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
    return make_settings(openrouter_model="or-model", llm_fallback_model="or-fallback", **kw)


def test_no_override_uses_the_engine_defaults():
    settings = cfg()
    agent = agent_in()
    assert llm_model_for_agent(agent, settings) == "or-model"
    assert llm_fallback_model_for_agent(agent, settings) == "or-fallback"


def test_workspace_models_override_the_defaults():
    settings = cfg()
    agent = agent_in(llm_model="google/gemini-2.5-flash-lite", llm_fallback_model="openai/gpt-4o-mini")
    assert llm_model_for_agent(agent, settings) == "google/gemini-2.5-flash-lite"
    assert llm_fallback_model_for_agent(agent, settings) == "openai/gpt-4o-mini"


def test_a_stored_groq_choice_is_ignored_and_its_model_is_never_sent_to_openrouter():
    settings = cfg()
    agent = agent_in(llm_provider="groq", llm_model="openai/gpt-oss-20b")
    assert effective_llm_provider(agent, settings) == "openrouter"
    assert llm_model_for_agent(agent, settings) == "or-model"


def test_a_fallback_equal_to_the_primary_is_no_fallback():
    agent = agent_in(llm_model="same", llm_fallback_model="same")
    assert llm_fallback_model_for_agent(agent, cfg()) is None


def test_resolve_llm_builds_a_failover_on_the_shared_openrouter_client():
    client = object()
    svc = Services(cfg(), {"openrouter": client, "groq": None})
    llm = resolve_llm(svc, agent_in(llm_fallback_model="ws-fallback"))
    assert isinstance(llm, FailoverLLM)
    assert llm._primary is client and llm._fallback is client and llm.fallback_model == "ws-fallback"


def test_persona_snapshot_carries_the_fallback_model():
    persona = PersonaIn(
        key="ab", name="A", language="en", instructions="i", greeting="g", version=1,
        llm_model="p-model", llm_fallback_model="f-model",
    )
    assert persona.llm_fallback_model == "f-model"


def test_same_primary_and_fallback_provider_is_still_rejected():
    with pytest.raises(ValueError, match="llm_fallback_provider must differ"):
        agent_in(llm_provider="groq", llm_fallback_provider="groq")
