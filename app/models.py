from datetime import datetime
from typing import Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.alias_generators import to_camel

from app.phone import indian_number_problem

Language = Literal["en", "hi", "ta", "te"]
Workflow = Literal["general", "lead_qualification", "faq", "appointment", "order"]

# Languages each supported ElevenLabs TTS model can speak (of the ones this platform offers).
# Telugu is only covered by eleven_v3: https://elevenlabs.io/docs/overview/models
TTS_MODEL_LANGUAGES: dict[str, set[str]] = {
    "eleven_flash_v2_5": {"en", "hi", "ta"},
    "eleven_multilingual_v2": {"en", "hi", "ta"},
    "eleven_v3": {"en", "hi", "ta", "te"},
}


BAKBAK_MODELS = {"standard", "m1"}


class AgentIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    language: Language
    workflow: Workflow = "general"
    instructions: str = Field(min_length=1, description="Persona, business context and goals.")
    greeting: str = Field(min_length=1, description="First sentence spoken when the call connects.")
    knowledge: str = Field("", description="Facts and FAQ answers the agent may use.")
    collect_fields: list[str] = Field(
        default_factory=list,
        description="Details to gather (lead qualification / appointment), e.g. name, budget.",
    )
    voice_id: str | None = Field(
        None, description="Overrides the language's default voice. With Bakbak, set together with tts_model_id."
    )
    tts_model_id: str | None = Field(
        None, description="Overrides the default TTS model (Bakbak: standard or m1, the voice_id's model)."
    )
    llm_model: str | None = Field(
        None,
        description=(
            "Overrides the primary model (OPENROUTER_MODEL) for this agent. The fallback always uses GROQ_MODEL."
        ),
    )
    is_inbound_default: bool = Field(False, description="Answer inbound calls with this agent.")
    client_id: str | None = Field(
        None, max_length=64, description="Dashboard workspace this agent belongs to (campaign dispatch)."
    )
    persona_key: str | None = Field(
        None,
        pattern=r"^[a-z0-9-]{2,40}$",
        description="Dashboard persona this agent serves. Unique per client_id; requires client_id.",
    )

    @model_validator(mode="after")
    def _persona_needs_client(self) -> Self:
        if self.persona_key and not self.client_id:
            raise ValueError("persona_key requires client_id")
        if self.is_inbound_default and self.client_id:
            raise ValueError(
                "an agent that belongs to a workspace cannot be the engine-wide inbound default; "
                "publish an inbound route for the number from the dashboard to choose its persona"
            )
        return self


class Agent(AgentIn):
    id: int
    created_at: datetime
    updated_at: datetime


class OutboundCallIn(BaseModel):
    agent_id: int
    to: str = Field(pattern=r"^\+?[0-9]{10,15}$", description="Number to dial, e.g. +919876543210.")
    contact_name: str | None = Field(
        None, max_length=100, description="Who is being called; fills [[CALLER_CONTEXT]] in the agent prompt."
    )

    @field_validator("contact_name")
    @classmethod
    def _single_line(cls, value: str | None) -> str | None:
        return " ".join(value.split()) or None if value else None


# ---- campaign dispatch (the dashboard's contract; JSON keys are camelCase) ---------------------


class _Camel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


_CLOCK = r"^([01][0-9]|2[0-3]):[0-5][0-9]$"


class DispatchLead(_Camel):
    id: str = Field(min_length=1, max_length=64)
    phone: str = Field(pattern=r"^\+?[0-9]{10,15}$")
    # The dashboard sends an em dash when a lead has no name.
    name: str | None = Field(None, max_length=100)

    @field_validator("name")
    @classmethod
    def _contact_name(cls, value: str | None) -> str | None:
        cleaned = " ".join(value.split()) if value else ""
        return None if cleaned in ("", "—") else cleaned


class CallingWindow(_Camel):
    start: str = Field(pattern=_CLOCK)
    end: str = Field(pattern=_CLOCK)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end <= self.start:
            raise ValueError("callingWindow.end must be after callingWindow.start")
        return self


class RetryPolicy(_Camel):
    max_tries: int = Field(ge=1, le=20)
    retry_gap_minutes: int = Field(ge=1, le=10_080)
    callback_window_days: int = Field(ge=1, le=365)


class PersonaIn(_Camel):
    """What the dashboard's persona says, as it was when the campaign was started."""

    key: str = Field(pattern=r"^[a-z0-9-]{2,40}$")
    name: str = Field(min_length=1, max_length=100)
    language: Language
    workflow: Workflow = "general"
    instructions: str = Field(min_length=1)
    greeting: str = Field(min_length=1)
    knowledge: str = ""
    version: int = Field(ge=1)


class DispatchIn(_Camel):
    idempotency_key: str = Field(min_length=8, max_length=100)
    campaign_id: str = Field(min_length=1, max_length=64)
    client_id: str = Field(min_length=1, max_length=64)
    persona: PersonaIn
    calling_window: CallingWindow
    retry_policy: RetryPolicy
    leads: list[DispatchLead] = Field(min_length=1, max_length=5000)
    webhook_url: str = Field(max_length=500)

    @field_validator("webhook_url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError("webhookUrl must be an http(s) URL")
        return value

    @model_validator(mode="after")
    def _distinct_leads(self) -> Self:
        ids = [lead.id for lead in self.leads]
        if len(set(ids)) != len(ids):
            raise ValueError("lead ids must be unique")
        return self


class InboundRoutePublishIn(_Camel):
    """One published inbound route: the number, who owns it, and the persona that answers it."""

    phone: str = Field(pattern=r"^\+[1-9][0-9]{9,14}$", description="E.164, for example +918047491899.")
    client_id: str = Field(min_length=1, max_length=64)
    version: int = Field(ge=1, description="Per-number publication counter; must increase on every publish.")
    active: bool
    persona: PersonaIn

    @field_validator("phone")
    @classmethod
    def _well_formed_indian_number(cls, value: str) -> str:
        if problem := indian_number_problem(value):
            raise ValueError(problem)
        return value


class InboundRouteOut(_Camel):
    phone: str
    client_id: str
    version: int
    active: bool
    persona_key: str
    persona_version: int
    workflow: Workflow
    status: Literal["applied", "unchanged", "current"]


class UnroutedOut(_Camel):
    """A dialled number that calls reached the engine on while no route was published for it."""

    phone: str
    raw_example: str
    calls: int
    first_seen: datetime
    last_seen: datetime
    answered_by: Literal["default_agent", "refused"]


class DispatchOut(_Camel):
    engine_job_id: str | None
    status: Literal["accepted", "rejected"]
    reason: str | None = None


JobStatus = Literal["queued", "running", "completed", "cancelled"]


class JobOut(_Camel):
    engine_job_id: str
    status: JobStatus
    calls_completed: int
    calls_failed: int


class CancelOut(_Camel):
    status: JobStatus


class CallTurn(BaseModel):
    idx: int
    user_text: str | None
    agent_text: str
    interrupted: bool
    endpointing_ms: int | None
    llm_ttft_ms: int | None
    tts_ttfb_ms: int | None
    response_latency_ms: int | None


class Call(BaseModel):
    id: int
    agent_id: int
    direction: Literal["inbound", "outbound"]
    exotel_call_sid: str | None
    from_number: str | None
    to_number: str | None
    contact_name: str | None
    status: str
    end_reason: str | None
    created_at: datetime
    started_at: datetime | None
    ended_at: datetime | None
    connected_seconds: float | None
    usage: dict[str, Any] | None
    cost: dict[str, Any] | None
    latency: dict[str, Any] | None
    outcome: str | None
    outcome_data: dict[str, Any] | None
    summary: str | None


class CallDetail(Call):
    turns: list[CallTurn]
