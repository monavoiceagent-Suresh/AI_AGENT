import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from app.models import Agent

# India has no daylight saving, so a fixed offset is exact and avoids a tz database dependency.
IST = timezone(timedelta(hours=5, minutes=30), name="IST")
PLACEHOLDER = re.compile(r"\[\[(KNOWLEDGE|CALLER_CONTEXT|CURRENT_DATETIME)\]\]")

END_CALL_TOKEN = "<end_call>"

LANGUAGE_RULES = {
    "en": "Reply in English (Indian English is fine).",
    "hi": "Reply in natural spoken Hindi written in Devanagari script; common English words are fine.",
    "ta": "Reply in natural spoken Tamil written in Tamil script; common English words are fine.",
    "te": "Reply in natural spoken Telugu written in Telugu script; common English words are fine.",
}

# Spoken when the LLM or TTS fails mid-conversation, so the caller is never left in silence.
FALLBACK_PHRASES = {
    "en": "Sorry, I'm having a little trouble right now. Could you please say that again?",
    "hi": "माफ़ कीजिए, मुझे अभी थोड़ी दिक्कत हो रही है। क्या आप दोबारा बता सकते हैं?",
    "ta": "மன்னிக்கவும், எனக்கு இப்போது சிறிய சிக்கல் உள்ளது. தயவுசெய்து மீண்டும் சொல்ல முடியுமா?",
    "te": "క్షమించండి, నాకు ఇప్పుడు కొంచెం ఇబ్బందిగా ఉంది. దయచేసి మళ్లీ చెప్పగలరా?",
}

WORKFLOW_GOALS = {
    "general": "Help the caller with their request.",
    "lead_qualification": (
        "Qualify the caller as a sales lead. Naturally collect the details listed below, one question at a "
        "time, and confirm anything unclear."
    ),
    "faq": (
        "Answer the caller's questions using only the knowledge below. If the answer is not there, say so "
        "and offer a callback from the team. Never invent facts."
    ),
    "order": (
        "Take a grocery or retail order, or answer the caller's enquiry. For an order, collect the details "
        "listed below one at a time (items with quantities and units, brand if they care), then read the whole "
        "order back to confirm. Never state or promise prices, stock, offers, delivery charges or delivery "
        "times; say the store will confirm them."
    ),
    "appointment": (
        "Take an appointment request. Collect the details listed below one at a time, then read the request "
        "back to confirm. Do not promise a confirmed slot; say the team will confirm it."
    ),
}

OUTCOMES = {
    "general": ["resolved", "unresolved", "callback_requested", "no_conversation"],
    "lead_qualification": ["qualified", "not_qualified", "needs_follow_up", "no_conversation"],
    "faq": ["answered", "unanswered", "callback_requested", "no_conversation"],
    "appointment": ["appointment_requested", "declined", "needs_follow_up", "no_conversation"],
    "order": ["order_placed", "enquiry_answered", "callback_requested", "no_conversation"],
}


def caller_context(direction: str, contact_name: str | None) -> str:
    """Who is on the call, for the [[CALLER_CONTEXT]] placeholder (no phone numbers are sent to the LLM)."""
    if direction == "inbound":
        return "Inbound call; the caller's name is not known."
    if contact_name:
        return f"Outbound call to {contact_name}."
    return "Outbound call; the contact's name is not known."


def system_prompt(agent: Agent, context: str, now: datetime | None = None) -> str:
    """Build the system prompt, filling the [[KNOWLEDGE]], [[CALLER_CONTEXT]] and [[CURRENT_DATETIME]] placeholders."""
    knowledge = agent.knowledge.strip()
    values = {
        "KNOWLEDGE": knowledge or "(no knowledge provided)",
        "CALLER_CONTEXT": context,
        "CURRENT_DATETIME": (now or datetime.now(IST)).astimezone(IST).strftime("%A, %d %B %Y, %I:%M %p IST"),
    }
    parts = [
        PLACEHOLDER.sub(lambda m: values[m.group(1)], agent.instructions.strip()),
        f"Goal: {WORKFLOW_GOALS[agent.workflow]}",
    ]
    if agent.collect_fields:
        parts.append("Details to collect: " + ", ".join(agent.collect_fields) + ".")
    if knowledge and "[[KNOWLEDGE]]" not in agent.instructions:
        parts.append("Knowledge:\n" + knowledge)
    parts.append(
        "This is a live phone call and everything you write is spoken aloud. "
        f"{LANGUAGE_RULES[agent.language]} Keep each reply to one or two short sentences and ask one "
        "question at a time. Do not use lists, markdown, emojis or symbols. Write numbers, dates, times and "
        "amounts as words the way they should be spoken. If you did not catch something, ask the caller to "
        "repeat it. "
        f"When the conversation is finished, say a brief goodbye and end your reply with {END_CALL_TOKEN}."
    )
    return "\n\n".join(parts)


def outcome_messages(agent: Agent, transcript: list[tuple[str, str]]) -> list[dict[str, str]]:
    lines = "\n".join(f"{speaker}: {text}" for speaker, text in transcript)
    fields = agent.collect_fields or []
    schema = {
        "outcome": " | ".join(OUTCOMES[agent.workflow]),
        "fields": {name: "value stated by the caller, or null" for name in fields},
        "summary": "two-sentence English summary of the call",
    }
    return [
        {
            "role": "system",
            "content": "You analyse phone call transcripts and reply with a single JSON object only.",
        },
        {
            "role": "user",
            "content": (
                f"Call goal: {WORKFLOW_GOALS[agent.workflow]}\n\nTranscript:\n{lines}\n\n"
                f"Return JSON with exactly this shape:\n{json.dumps(schema, ensure_ascii=False)}"
            ),
        },
    ]


def parse_outcome(agent: Agent, raw: str) -> tuple[str, dict[str, Any], str]:
    """Return (outcome, fields, summary); malformed model output yields outcome 'extraction_failed'."""
    start, end = raw.find("{"), raw.rfind("}")
    try:
        data = json.loads(raw[start : end + 1]) if start != -1 else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict) or data.get("outcome") not in OUTCOMES[agent.workflow]:
        return "extraction_failed", {}, ""
    fields = data.get("fields") if isinstance(data.get("fields"), dict) else {}
    return data["outcome"], fields, str(data.get("summary") or "")
