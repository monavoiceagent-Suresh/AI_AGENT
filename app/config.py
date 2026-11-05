from functools import lru_cache
from typing import Literal, Self

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.interfaces import Voice

TTSProvider = Literal["bakbak", "elevenlabs_premium", "elevenlabs_standard", "deepgram_aura"]
ASRProvider = Literal["elevenlabs_scribe", "deepgram_nova3"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000
    # Bearer token required by the management API (agents, calls).
    app_secret_key: SecretStr
    # Public HTTPS base URL that Exotel can reach (tunnel in local development).
    public_base_url: str

    exotel_account_sid: str
    exotel_api_key: SecretStr
    exotel_api_token: SecretStr
    exotel_api_base_url: str = "https://api.exotel.com"
    exotel_caller_id: str
    # App Bazaar flow ID containing the Voicebot applet; required for outbound calls.
    exotel_app_id: str = ""
    # Shared secret appended to the StatusCallback URL (Exotel does not sign webhooks).
    exotel_webhook_secret: SecretStr

    enable_inbound_calls: bool = True
    enable_outbound_calls: bool = False
    max_concurrent_calls: int = 1
    max_call_duration_seconds: int = 1800

    elevenlabs_api_key: SecretStr
    elevenlabs_asr_model_id: str = "scribe_v2_realtime"
    # Standard (low-latency) tier model; Telugu is not covered by Flash, only by the Premium tier (eleven_v3).
    elevenlabs_tts_model_id: str = "eleven_flash_v2_5"
    elevenlabs_premium_model_id: str = "eleven_v3"
    # Needed only when ElevenLabs is a configured TTS provider (primary or fallback, either tier).
    elevenlabs_tts_voice_id: str = ""
    elevenlabs_tts_stability: float = 0.5
    elevenlabs_tts_similarity_boost: float = 0.75

    deepgram_api_key: SecretStr | None = None
    # Aura has no Hindi/Tamil/Telugu voices; this must be an "-en" voice (see app/providers/deepgram_tts.py).
    deepgram_tts_voice: str = "aura-2-thalia-en"

    # Independent primary/fallback choice for ASR and TTS. The fallback is used only when its own credentials
    # are configured; None disables that fallback entirely. Primary and fallback must differ.
    asr_provider: ASRProvider = "elevenlabs_scribe"
    asr_fallback_provider: ASRProvider | None = "deepgram_nova3"
    tts_provider: TTSProvider = "elevenlabs_premium"
    tts_fallback_provider: TTSProvider | None = "bakbak"

    bakbak_api_key: SecretStr | None = None
    bakbak_base_url: str = "https://hub.getraya.app/v1"
    # Agent language (en, hi, ta, te) -> voice and model. Voice IDs are tied to one model and one language;
    # list them with GET {BAKBAK_BASE_URL}/voices.
    bakbak_voices: dict[str, Voice] = Field(default_factory=dict)

    # Which LLM answers first (any letter case). The other one, if its key is set, is the automatic fallback.
    llm_provider: Literal["openrouter", "groq"] = "openrouter"

    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # OPENROUTER_FALLBACK_MODEL is the name this setting had when OpenRouter was always the fallback; it still works.
    openrouter_model: str | None = Field(
        None, validation_alias=AliasChoices("openrouter_model", "openrouter_fallback_model")
    )
    openrouter_max_output_tokens: int = 250
    openrouter_temperature: float = 0.4

    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "openai/gpt-oss-20b"
    # Reasoning tokens count against this budget, so it must cover the thinking plus the spoken reply.
    groq_max_output_tokens: int = 1024
    groq_temperature: float = 0.4

    # Timeout and retries per provider. Unset means the default for its role: a primary gives up quickly (4 s, no
    # retry, because every retry adds a full timeout before the fallback gets a turn); a fallback has nothing behind
    # it, so it gets 8 s and one retry.
    openrouter_timeout_ms: int | None = None
    openrouter_max_retries: int | None = None

    # Safety limits against excessive LLM requests, shared by whichever provider is primary or fallback.
    # A provider's per-minute limit is per PROCESS (shared by every concurrent call, so it is a real global cap);
    # the request limit is per PHONE CALL. Any of these set to 0 or less disables that particular limit.
    # OpenRouter's own free-tier models cap at 20 requests/minute; 18 leaves headroom before they would 429 us.
    openrouter_max_requests_per_minute: int = 18
    groq_max_requests_per_minute: int = 60
    # A generous backstop against a stuck conversation or a run of failures, not a normal-call limit.
    llm_max_requests_per_call: int = 40
    llm_retry_backoff_base_ms: int = 300
    llm_retry_backoff_max_ms: int = 2000
    # After this many consecutive failed responses the agent ends the call instead of trying forever.
    max_consecutive_llm_failures: int = 3
    # A committed transcript identical to the one just answered, delivered again within this window, is treated
    # as a duplicate delivery (an ASR/websocket retransmit) and dropped instead of starting a second response.
    llm_dedup_window_ms: int = 700

    groq_timeout_ms: int | None = None
    groq_max_retries: int | None = None

    database_url: str
    database_pool_min_size: int = 1
    database_pool_max_size: int = 5

    # Required: the M2 channel allocator (app/allocator.py) has no in-memory fallback. A missing
    # value fails the engine at startup with a clear error, rather than silently reverting to the
    # old per-process capacity accounting this replaces.
    redis_url: str
    redis_connect_timeout_seconds: float = Field(5.0, gt=0)
    # Namespaces the allocator's Redis hash key, so more than one engine deployment (or a test
    # suite) can share one Redis without their reservations colliding.
    redis_key_prefix: str = "allocator"
    # Compatibility setting only: reservations no longer expire without terminal proof.
    allocation_reservation_seconds: int = Field(90, gt=0)

    session_idle_timeout_seconds: int = 120
    # Silence after speech before Scribe commits a transcript (end of user turn).
    endpointing_min_silence_ms: int = 300
    barge_in_enabled: bool = True
    # Max audio queued ahead of playback at Exotel; bounds what an interruption must discard.
    audio_queue_max_ms: int = 2000
    asr_timeout_ms: int = 5000
    tts_timeout_ms: int = 5000

    usd_to_inr: float = 85.0
    elevenlabs_asr_usd_per_hour: float = 0.39
    elevenlabs_tts_usd_per_1k_chars: dict[str, float] = Field(
        default_factory=lambda: {
            "eleven_flash_v2_5": 0.04,
            "eleven_multilingual_v2": 0.08,
            "eleven_v3": 0.08,
        }
    )
    # Bakbak publishes no list price: set your contracted rate, or TTS cost is left unpriced.
    bakbak_inr_per_1k_chars: float | None = None
    # Exotel per-minute rate from your plan; telephony cost is omitted when unset.
    exotel_inr_per_minute: float | None = None
    target_cost_per_connected_minute_inr: float = 3.0

    log_transcripts: bool = False

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _llm_provider_any_case(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_llm_provider(self) -> Self:
        if self.llm_provider == "groq" and self.groq_api_key is None:
            raise ValueError("LLM_PROVIDER=groq requires GROQ_API_KEY")
        if self.llm_provider == "openrouter" and self.openrouter_api_key is None:
            raise ValueError("LLM_PROVIDER=openrouter requires OPENROUTER_API_KEY")
        if self.openrouter_api_key is not None and not self.openrouter_model:
            raise ValueError("OPENROUTER_API_KEY is set, so OPENROUTER_MODEL is required too")
        return self

    @property
    def primary_llm_model(self) -> str:
        """Model used when an agent doesn't set llm_model: the primary provider's."""
        return self.groq_model if self.llm_provider == "groq" else self.openrouter_model

    # Campaign dispatch from the Mona dashboard. The same value must be the dashboard's
    # VOICE_ENGINE_WEBHOOK_SECRET; without it dispatches are rejected because results could not be reported.
    dashboard_webhook_secret: SecretStr | None = None
    # Where finished inbound calls are reported (the dashboard's /api/webhooks/voice-engine). Inbound calls
    # belong to no dispatch job that could carry a URL. Unset = inbound calls are not reported.
    dashboard_webhook_url: str | None = None
    # A report that still cannot be delivered after this many attempts is given up on (and kept for inspection).
    webhook_max_attempts: int = Field(12, ge=1)
    dispatcher_poll_seconds: float = Field(2.0, gt=0)

    # Slots kept free for inbound calls: outbound and campaign calls may use only
    # MAX_CONCURRENT_CALLS - INBOUND_RESERVED_CALLS in total. 0 keeps today's behaviour.
    inbound_reserved_calls: int = Field(0, ge=0)
    # Answer calls to numbers with no published route with the engine-wide default agent (an agent with
    # is_inbound_default and no client_id). Turn off once every number is routed from the dashboard.
    inbound_legacy_default_enabled: bool = True
    # When the stream's `to` matches no route, read Exotel's own record of the call (read-only, the account's
    # credentials) and try the ExoPhone that was dialled. Exotel's stream `to` is not always that number.
    exotel_resolve_called_number: bool = True
    exotel_lookup_timeout_seconds: float = Field(3.0, gt=0, le=10)

    # Stage of the engine's own tenant validation (app/tenants.py): "log_only" never blocks a
    # request over an unrecognised/suspended client_id, only logs it; "enforcing" rejects it.
    # Flip only after Mona has backfilled and is keeping every existing tenant synced (PUT /tenants).
    tenant_validation_mode: Literal["log_only", "enforcing"] = "log_only"

    @model_validator(mode="after")
    def _check_webhook_url(self) -> Self:
        if self.dashboard_webhook_url is not None and not self.dashboard_webhook_url.startswith(
            ("http://", "https://")
        ):
            raise ValueError("DASHBOARD_WEBHOOK_URL must be an http(s) URL")
        return self

    @property
    def inbound_report_url(self) -> str | None:
        """Inbound calls are reported only when both the destination and the signing secret are configured."""
        return self.dashboard_webhook_url if self.dashboard_webhook_secret is not None else None

    def webhook_configuration_warnings(self) -> list[str]:
        """Actionable startup warnings for a half-configured or missing dashboard webhook setup.

        Neither half is required to start the engine (a fresh install legitimately has neither yet, and
        POST /calls and /calls/test need no webhook at all), so these are warnings, not startup errors.
        """
        warnings: list[str] = []
        if self.enable_outbound_calls and self.dashboard_webhook_secret is None:
            warnings.append(
                "ENABLE_OUTBOUND_CALLS is true but DASHBOARD_WEBHOOK_SECRET is not set: every campaign "
                "dispatch will be rejected (POST /calls and /calls/test are unaffected; they need no "
                "webhook secret). Set DASHBOARD_WEBHOOK_SECRET to the same value as the dashboard's "
                "VOICE_ENGINE_WEBHOOK_SECRET."
            )
        if self.dashboard_webhook_url is not None and self.dashboard_webhook_secret is None:
            warnings.append(
                "DASHBOARD_WEBHOOK_URL is set but DASHBOARD_WEBHOOK_SECRET is not: finished inbound calls "
                "will never be reported to the dashboard (inbound_report_url is None without it)."
            )
        if self.dashboard_webhook_secret is not None and self.dashboard_webhook_url is None:
            warnings.append(
                "DASHBOARD_WEBHOOK_SECRET is set but DASHBOARD_WEBHOOK_URL is not: finished inbound calls "
                "will never be reported to the dashboard. Campaign dispatch is unaffected — its webhook "
                "URL comes from each dispatch request, not from this setting."
            )
        return warnings

    @model_validator(mode="after")
    def _check_inbound_reserve(self) -> Self:
        if self.inbound_reserved_calls > self.max_concurrent_calls:
            raise ValueError("INBOUND_RESERVED_CALLS cannot exceed MAX_CONCURRENT_CALLS")
        return self

    def tts_credentials_present(self, provider: TTSProvider) -> bool:
        """Whether the given provider's own credentials are configured, regardless of its role."""
        if provider == "bakbak":
            return self.bakbak_api_key is not None and bool(self.bakbak_voices)
        if provider in ("elevenlabs_premium", "elevenlabs_standard"):
            return bool(self.elevenlabs_tts_voice_id)
        return self.deepgram_api_key is not None  # deepgram_aura

    @model_validator(mode="after")
    def _check_tts_provider(self) -> Self:
        # Only the primary must be usable at startup; an unusable fallback is simply not built (see main.py),
        # exactly like the existing LLM primary/fallback pattern.
        if self.tts_fallback_provider == self.tts_provider:
            raise ValueError("TTS_FALLBACK_PROVIDER must differ from TTS_PROVIDER")
        if not self.tts_credentials_present(self.tts_provider):
            raise ValueError(f"TTS_PROVIDER={self.tts_provider} is missing its required credentials")
        return self

    @model_validator(mode="after")
    def _check_asr_provider(self) -> Self:
        if self.asr_fallback_provider == self.asr_provider:
            raise ValueError("ASR_FALLBACK_PROVIDER must differ from ASR_PROVIDER")
        if self.asr_provider == "deepgram_nova3" and self.deepgram_api_key is None:
            raise ValueError("ASR_PROVIDER=deepgram_nova3 requires DEEPGRAM_API_KEY")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
