"""Call Outcome Rules validation (AgentIn/PersonaIn) and the backward-compatible with_persona merge."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.calls import with_persona
from app.models import Agent, AgentIn, PersonaIn

BASE_AGENT = dict(name="A", language="en", instructions="x", greeting="y")
BASE_PERSONA = dict(key="pp", name="A", language="en", instructions="x", greeting="y", version=1)
RULES = [{"name": "Interested", "definition": "Clear positive response."}]


def test_outcome_fallback_is_required_once_there_are_custom_rules():
    with pytest.raises(ValidationError, match="outcome_fallback is required"):
        AgentIn(**BASE_AGENT, outcome_rules=RULES)
    AgentIn(**BASE_AGENT, outcome_rules=RULES, outcome_fallback="No decision.")  # does not raise


def test_rule_names_must_be_unique_case_insensitively():
    dup = [*RULES, {"name": "interested", "definition": "Another one."}]
    with pytest.raises(ValidationError, match="must be unique"):
        AgentIn(**BASE_AGENT, outcome_rules=dup, outcome_fallback="No decision.")


def test_inconclusive_is_reserved_and_cannot_be_a_named_rule():
    rules = [{"name": "Inconclusive", "definition": "x"}]
    with pytest.raises(ValidationError, match="always-last fallback"):
        AgentIn(**BASE_AGENT, outcome_rules=rules, outcome_fallback="No decision.")


def test_persona_in_has_the_same_validation_as_agent_in():
    with pytest.raises(ValidationError, match="outcomeFallback is required"):
        PersonaIn(**BASE_PERSONA, outcomeRules=RULES)
    PersonaIn(**BASE_PERSONA, outcomeRules=RULES, outcomeFallback="No decision.")  # does not raise


def test_no_rules_needs_no_fallback():
    AgentIn(**BASE_AGENT)  # does not raise: outcome_rules defaults to [], outcome_fallback to None


def _agent() -> Agent:
    now = datetime.now(UTC)
    return Agent(**BASE_AGENT, id=1, created_at=now, updated_at=now)


def test_with_persona_fills_in_defaults_for_a_route_or_job_saved_before_outcome_rules_existed():
    # A dict with none of the newer keys at all, exactly like a persona snapshot stored before this feature.
    old_style_persona = {
        "name": "A",
        "language": "en",
        "workflow": "general",
        "instructions": "x",
        "greeting": "y",
        "knowledge": "",
    }

    agent = with_persona(_agent(), old_style_persona)

    assert agent.outcome_rules == [] and agent.outcome_fallback is None


def test_with_persona_applies_stored_custom_rules_as_real_objects_not_raw_dicts():
    persona = {
        "name": "A",
        "language": "en",
        "workflow": "general",
        "instructions": "x",
        "greeting": "y",
        "knowledge": "",
        "outcome_rules": RULES,
        "outcome_fallback": "No decision.",
    }

    agent = with_persona(_agent(), persona)

    assert agent.outcome_rules[0].name == "Interested"  # attribute access, not dict-key access
    assert agent.outcome_fallback == "No decision."
