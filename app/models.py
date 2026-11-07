from datetime import datetime
from typing import Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.alias_generators import to_camel

from app.config import ASRProvider, TTSProvider
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


class OutcomeRuleIn(BaseModel):
    """One named outcome in a persona's own classification vocabulary (the dashboard's Call Outcome
    Rules page). When a persona has any of these, outcome extraction classifies into these names
    instead of the fixed per-workflow list, so the dashboard's own wording comes back directly."""

    name: str = Field(min_length=2, max_length=60)
    definition: str = Field(min_length=1, max_length=600, description="When this outcome applies.")


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
    outcome_rules: list[OutcomeRuleIn] = Field(
        default_factory=list,
        max_length=10,
        description="Custom outcome vocabulary; empty uses the fixed per-workflow list instead.",
    )
    outcome_fallback: str | None = Field(
        None,
        max_length=600,
        description="Definition for 'Inconclusive', the always-last catch-all. Required if outcome_rules is set.",
    )
    voice_id: str | None = Field(
        None, description="Overrides the language's default voice. With Bakbak, set together with tts_model_id."
    )
    tts_model_id: str | None = Field(
        None, description="Overrides the default TTS model (Bakbak: standard or m1, the voice_id's model)."
    )
    voice_settings: dict | None = Field(
        None, description="ElevenLabs stability/similarity_boost/style overrides for voice_id. Only ElevenLabs reads this."
    )
    llm_model: str | None = Field(
        None,
        description=(
            "Overrides the primary model (OPENROUTER_MODEL) for this agent. The fallback always uses GROQ_MODEL."
        ),
    )
    asr_provider: ASRProvider | None = Field(None, description="Overrides ASR_PROVIDER for this agent.")
    asr_fallback_provider: ASRProvider | None = Field(
        None, description="Overrides ASR_FALLBACK_PROVIDER for this agent."
    )
    tts_provider: TTSProvider | None = Field(None, description="Overrides TTS_PROVIDER for this agent.")
    tts_fallback_provider: TTSProvider | None = Field(
        None, description="Overrides TTS_FALLBACK_PROVIDER for this agent."
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

    @model_validator(mode="after")
    def _check_outcome_rules(self) -> Self:
        if self.outcome_rules and not self.outcome_fallback:
            raise ValueError("outcome_fallback is required when outcome_rules is set")
        names = [r.name.strip().lower() for r in self.outcome_rules]
        if len(names) != len(set(names)):
            raise ValueError("outcome_rules names must be unique")
        if "inconclusive" in names:
            raise ValueError("'Inconclusive' is the always-last fallback; it cannot be a named rule")
        return self

    @model_validator(mode="after")
    def _check_provider_overrides(self) -> Self:
        if self.asr_provider is not None and self.asr_provider == self.asr_fallback_provider:
            raise ValueError("asr_fallback_provider must differ from asr_provider")
        if self.tts_provider is not None and self.tts_provider == self.tts_fallback_provider:
            raise ValueError("tts_fallback_provider must differ from tts_provider")
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
    attempts: int = Field(0, ge=0, le=20)
    next_attempt_at: datetime | None = None

    @field_validator("next_attempt_at")
    @classmethod
    def _aware_attempt(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("nextAttemptAt must include a timezone")
        return value

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
    # The workspace's own Call Outcome Rules, current when this was sent. Empty means the engine
    # classifies into its own fixed per-workflow list, exactly as before these fields existed.
    outcome_rules: list[OutcomeRuleIn] = Field(default_factory=list, max_length=10)
    outcome_fallback: str | None = Field(None, max_length=600)
    # The dashboard's own resolved Persona -> Workspace -> Global provider choice. Omitted/null at every
    # level means the engine's own global default (Settings) is used, exactly as before these fields existed.
    asr_provider: ASRProvider | None = None
    asr_fallback_provider: ASRProvider | None = None
    tts_provider: TTSProvider | None = None
    tts_fallback_provider: TTSProvider | None = None

    @model_validator(mode="after")
    def _check_outcome_rules(self) -> Self:
        if self.outcome_rules and not self.outcome_fallback:
            raise ValueError("outcomeFallback is required when outcomeRules is set")
        names = [r.name.strip().lower() for r in self.outcome_rules]
        if len(names) != len(set(names)):
            raise ValueError("outcomeRules names must be unique")
        if "inconclusive" in names:
            raise ValueError("'Inconclusive' is the always-last fallback; it cannot be a named rule")
        return self

    @model_validator(mode="after")
    def _check_provider_overrides(self) -> Self:
        if self.asr_provider is not None and self.asr_provider == self.asr_fallback_provider:
            raise ValueError("asrFallbackProvider must differ from asrProvider")
        if self.tts_provider is not None and self.tts_provider == self.tts_fallback_provider:
            raise ValueError("ttsFallbackProvider must differ from ttsProvider")
        return self


class VoiceSettingsIn(_Camel):
    """ElevenLabs' own expressiveness controls for a chosen voice. Any field left unset keeps the
    engine's fixed default for that field — this never applies to non-ElevenLabs providers."""

    stability: float | None = Field(None, ge=0, le=1)
    similarity_boost: float | None = Field(None, ge=0, le=1)
    style: float | None = Field(None, ge=0, le=1)


class VoiceChoice(_Camel):
    """A campaign's (or test call's) own voice choice, from the real voices GET /voices lists. Overrides
    whatever the agent for this persona already has; set both or neither, like AgentIn's voice_id/tts_model_id."""

    voice_id: str | None = Field(None, min_length=1)
    tts_model_id: str | None = Field(None, min_length=1)
    voice_settings: VoiceSettingsIn | None = None


class TestCallIn(VoiceChoice):
    """One persona-pinned outbound test; Mona requests a durable, reported job with a request key.

    Legacy management callers without that key keep the original immediate-call response.
    """

    client_id: str = Field(min_length=1, max_length=64)
    persona: PersonaIn
    to: str = Field(pattern=r"^\+?[0-9]{10,15}$", description="Number to dial, e.g. +919876543210.")
    contact_name: str | None = Field(None, max_length=100)
    # Mona opts into the existing durable dispatcher; legacy management clients still dial directly.
    idempotency_key: str | None = Field(None, min_length=8, max_length=100)
    webhook_url: str | None = Field(None, max_length=500)

    @model_validator(mode="after")
    def _reporting_pair(self) -> Self:
        if bool(self.idempotency_key) != bool(self.webhook_url):
            raise ValueError("idempotencyKey and webhookUrl must be supplied together")
        return self

    @field_validator("contact_name")
    @classmethod
    def _single_line(cls, value: str | None) -> str | None:
        return " ".join(value.split()) or None if value else None


class DispatchIn(VoiceChoice):
    idempotency_key: str = Field(min_length=8, max_length=100)
    campaign_id: str = Field(min_length=1, max_length=64)
    client_id: str = Field(min_length=1, max_length=64)
    persona: PersonaIn
    calling_window: CallingWindow
    retry_policy: RetryPolicy
    leads: list[DispatchLead] = Field(min_length=1, max_length=5000)
    webhook_url: str = Field(max_length=500)
    # A UTC instant. Existing next_attempt_at and the India calling window enforce both constraints.
    not_before: datetime | None = None
    # The outbound caller ID Mona resolved and validated for this tenant/campaign at dispatch time
    # (M1 number inventory). Persisted on the job as a snapshot (app/dispatch_db.py create_job): once
    # accepted, a job keeps this value even if Mona's own assignment later changes. None means the
    # caller did not resolve one (a job created before this field existed, or a legacy caller), so
    # calls keep using the engine-wide EXOTEL_CALLER_ID, exactly as before this field existed.
    caller_id: str | None = Field(None, pattern=r"^\+[1-9][0-9]{9,14}$")

    @field_validator("not_before")
    @classmethod
    def _aware_start(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("notBefore must include a timezone")
        return value

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
        if any(lead.attempts >= self.retry_policy.max_tries for lead in self.leads):
            raise ValueError("leads must have attempts remaining")
        return self


class TenantCapacityOut(_Camel):
    """One tenant's share of the account-wide channel budget, for the Super Admin capacity view."""

    client_id: str
    connected: int
    reserved: int
    queued_leads: int


class CapacityOut(_Camel):
    """`GET /capacity`: the M2 allocator's live state, for Super Admin operational visibility (M3)."""

    max_concurrent_calls: int
    inbound_reserved_calls: int
    occupied: int
    connected: int
    reserved: int
    inbound_occupied: int
    outbound_occupied: int
    unattributed: int = 0
    # Rows whose state suggests a stuck/lost transition — never expected in healthy operation; a
    # non-zero count here is worth an operator's attention (see app/monitoring.py).
    stale_reserved: int
    stuck_releasing: int
    tenants: list[TenantCapacityOut]


class TenantSyncIn(_Camel):
    """`PUT /tenants`: Mona's own record of one tenant, mirrored here for the engine's own
    client_id validation (app/tenants.py). Not a second source of truth for tenant data."""

    id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=200)
    status: Literal["active", "suspended"]


class TenantOut(_Camel):
    id: str
    name: str
    status: Literal["active", "suspended"]


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


class VoiceOption(_Camel):
    """One voice the configured TTS provider actually has, as it reported it. Never hardcoded."""

    voice_id: str
    name: str
    language: str | None = None
    # Bakbak only: a voice belongs to exactly one model (standard/m1); ElevenLabs voices are model-independent.
    model_id: str | None = None


class VoicesOut(_Camel):
    provider: Literal["bakbak", "elevenlabs", "deepgram"]
    voices: list[VoiceOption]


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
