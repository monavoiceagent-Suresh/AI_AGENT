from datetime import UTC, datetime

from app import prompts
from app.metrics import call_cost, latency_summary, tts_inr_per_1k_chars
from app.models import Agent
from tests.fakes import make_settings as settings


def agent(**overrides) -> Agent:
    now = datetime.now(UTC)
    base = dict(
        id=1,
        name="Leads",
        language="ta",
        workflow="lead_qualification",
        instructions="You call on behalf of Acme Homes.",
        greeting="Vanakkam!",
        knowledge="Flats start at 45 lakh.",
        collect_fields=["name", "budget"],
        created_at=now,
        updated_at=now,
    )
    return Agent(**(base | overrides))


def test_cost_for_one_minute_call_on_elevenlabs():
    cfg = settings(exotel_inr_per_minute=0.6)
    cost = call_cost(
        cfg,
        connected_seconds=60,
        asr_seconds=60,
        tts_usage=[(400, tts_inr_per_1k_chars(cfg, "elevenlabs", "eleven_flash_v2_5"))],
        llm_cost_usd=0.0002,
    )
    # ASR 0.39/h -> 0.5525; TTS 0.4k chars * 0.04 USD -> 1.36; LLM 0.017; telephony 0.6
    assert cost["asr"] == 0.5525 and cost["tts"] == 1.36 and cost["llm"] == 0.017
    assert cost["total"] == 2.5295 and cost["per_connected_minute"] == 2.5295
    assert cost["within_target"] is True and cost["complete"] is True


def test_bakbak_cost_uses_the_configured_inr_rate():
    cfg = settings(bakbak_inr_per_1k_chars=2.5, exotel_inr_per_minute=0.6)
    rate = tts_inr_per_1k_chars(cfg, "bakbak", "m1")
    assert rate == 2.5
    cost = call_cost(cfg, connected_seconds=60, asr_seconds=0, tts_usage=[(400, rate)], llm_cost_usd=0)
    assert cost["tts"] == 1.0 and cost["complete"] is True


def test_cost_adds_fallback_provider_characters():
    cost = call_cost(
        settings(exotel_inr_per_minute=0),
        connected_seconds=60,
        asr_seconds=0,
        tts_usage=[(100, 5.0), (50, 3.4)],
        llm_cost_usd=0,
    )
    assert cost["tts"] == 0.67  # 0.1k * 5.0 + 0.05k * 3.4


def test_cost_marks_unpriced_tts_as_incomplete():
    # Bakbak publishes no list price: with no configured rate the TTS cost is unknown, not zero.
    cfg = settings()
    cost = call_cost(
        cfg,
        connected_seconds=30,
        asr_seconds=30,
        tts_usage=[(100, tts_inr_per_1k_chars(cfg, "bakbak", "m1")), (0, None)],
        llm_cost_usd=0,
    )
    assert cost["tts"] is None and cost["telephony"] is None and cost["complete"] is False
    assert tts_inr_per_1k_chars(cfg, "elevenlabs", "unknown_model") is None


def test_latency_summary():
    turns = [
        {
            "interrupted": False,
            "response_latency_ms": None,
            "llm_ttft_ms": None,
            "tts_ttfb_ms": 200,
            "endpointing_ms": None,
        },
        {
            "interrupted": True,
            "response_latency_ms": 900,
            "llm_ttft_ms": 400,
            "tts_ttfb_ms": 300,
            "endpointing_ms": 500,
        },
        {
            "interrupted": False,
            "response_latency_ms": 700,
            "llm_ttft_ms": 300,
            "tts_ttfb_ms": 250,
            "endpointing_ms": 450,
        },
    ]
    summary = latency_summary(turns)
    assert summary["turns"] == 3 and summary["interruptions"] == 1
    assert summary["response_ms_avg"] == 800 and summary["response_ms_p50"] == 700 and summary["response_ms_p95"] == 900
    assert summary["tts_ttfb_ms_avg"] == 250


def test_system_prompt_includes_workflow_language_and_knowledge():
    prompt = prompts.system_prompt(agent(), "ctx")
    assert "Acme Homes" in prompt and "45 lakh" in prompt
    assert "Details to collect: name, budget." in prompt
    assert "Tamil script" in prompt and prompts.END_CALL_TOKEN in prompt


def test_placeholders_are_filled_and_knowledge_is_not_duplicated():
    a = agent(
        instructions="Use only [[KNOWLEDGE]]. Consider [[CALLER_CONTEXT]] at [[CURRENT_DATETIME]]. Keep [[UNKNOWN]].",
        knowledge="Open 9 to 6",
    )
    now = datetime(2026, 10, 26, 10, 15, tzinfo=UTC)  # 15:45 in India
    prompt = prompts.system_prompt(a, "Outbound call to Ravi.", now)
    assert "Use only Open 9 to 6. Consider Outbound call to Ravi. at Monday, 26 October 2026, 03:45 PM IST." in prompt
    assert prompt.count("Open 9 to 6") == 1 and "Knowledge:" not in prompt
    assert "[[UNKNOWN]]" in prompt  # only the three documented placeholders are replaced


def test_placeholder_knowledge_falls_back_when_empty_and_plain_agents_are_unchanged():
    empty = agent(instructions="Facts: [[KNOWLEDGE]]", knowledge="  ")
    assert "Facts: (no knowledge provided)" in prompts.system_prompt(empty, "ctx")
    plain = prompts.system_prompt(agent(), "ctx")  # no placeholders: knowledge is appended as before
    assert "Knowledge:\nFlats start at 45 lakh." in plain and "ctx" not in plain


def test_caller_context_never_includes_phone_numbers():
    assert prompts.caller_context("inbound", None) == "Inbound call; the caller's name is not known."
    assert prompts.caller_context("outbound", "Ravi Kumar") == "Outbound call to Ravi Kumar."
    assert prompts.caller_context("outbound", None) == "Outbound call; the contact's name is not known."


def test_order_workflow_prompt_and_outcomes():
    a = agent(workflow="order", collect_fields=["items", "delivery_address"])
    prompt = prompts.system_prompt(a, "ctx")
    assert "read the whole order back" in prompt and "Details to collect: items, delivery_address." in prompt
    raw = '{"outcome": "order_placed", "fields": {"items": "2 kg rice", "delivery_address": null}, "summary": "Order."}'
    assert prompts.parse_outcome(a, raw)[0] == "order_placed"
    assert prompts.parse_outcome(a, '{"outcome": "qualified"}')[0] == "extraction_failed"


def test_parse_outcome():
    a = agent()
    raw = 'Sure:\n{"outcome": "qualified", "fields": {"name": "Ravi", "budget": "50 lakh"}, "summary": "Keen buyer."}'
    assert prompts.parse_outcome(a, raw) == ("qualified", {"name": "Ravi", "budget": "50 lakh"}, "Keen buyer.")
    assert prompts.parse_outcome(a, '{"outcome": "booked"}')[0] == "extraction_failed"
    assert prompts.parse_outcome(a, "no json")[0] == "extraction_failed"


def outcome_rules_agent(**overrides) -> Agent:
    rules = [
        {"name": "Interested", "definition": "The contact expresses a clear positive response."},
        {"name": "Not interested", "definition": "The contact clearly declines."},
    ]
    return agent(outcome_rules=rules, outcome_fallback="No clear decision was made.", **overrides)


def test_a_persona_with_custom_outcome_rules_is_classified_into_them_not_the_workflow_list():
    a = outcome_rules_agent()
    assert prompts.parse_outcome(a, '{"outcome": "Interested", "summary": "Keen."}')[0] == "Interested"
    assert prompts.parse_outcome(a, '{"outcome": "Inconclusive", "summary": "Unclear."}')[0] == "Inconclusive"
    # A fixed-workflow word that is not one of the persona's own rules no longer passes.
    assert prompts.parse_outcome(a, '{"outcome": "qualified", "summary": "x"}')[0] == "extraction_failed"


def test_custom_outcome_rules_appear_in_the_prompt_with_their_definitions_and_the_fallback():
    a = outcome_rules_agent()
    content = prompts.outcome_messages(a, [("Caller", "hi"), ("Agent", "hello")])[1]["content"]
    assert "The contact expresses a clear positive response." in content
    assert "The contact clearly declines." in content
    assert "No clear decision was made." in content
    assert '"outcome": "Interested | Not interested | Inconclusive"' in content
    assert "Call goal:" not in content  # the fixed per-workflow goal is replaced, not just appended


def test_without_custom_rules_the_fixed_per_workflow_list_is_used_exactly_as_before():
    a = agent()
    assert a.outcome_rules == [] and a.outcome_fallback is None
    content = prompts.outcome_messages(a, [("Caller", "hi")])[1]["content"]
    assert "Call goal:" in content
    assert prompts.parse_outcome(a, '{"outcome": "qualified", "summary": "x"}')[0] == "qualified"
