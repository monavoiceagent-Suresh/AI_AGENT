from functools import lru_cache
from typing import Literal, Self

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.interfaces import Voice


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
    elevenlabs_tts_model_id: str = "eleven_flash_v2_5"
    # Needed only when ElevenLabs is the TTS provider or the fallback.
    elevenlabs_tts_voice_id: str = ""
    elevenlabs_tts_stability: float = 0.5
    elevenlabs_tts_similarity_boost: float = 0.75

    tts_provider: Literal["bakbak", "elevenlabs"] = "bakbak"
    # Retry a sentence on ElevenLabs when the primary provider fails before producing audio.
    tts_fallback_enabled: bool = False
    bakbak_api_key: SecretStr | None = None
    bakbak_base_url: str = "https://hub.getraya.app/v1"
    # Agent language (en, hi, ta, te) -> voice and model. Voice IDs are tied to one model and one language;
    # list them with GET {BAKBAK_BASE_URL}/voices.
    bakbak_voices: dict[str, Voice] = Field(default_factory=dict)

    # Primary LLM: OpenRouter.
    openrouter_api_key: SecretStr
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # OPENROUTER_FALLBACK_MODEL is the name this setting had when OpenRouter was the fallback; it still works.
    openrouter_model: str = Field(validation_alias=AliasChoices("openrouter_model", "openrouter_fallback_model"))
    openrouter_max_output_tokens: int = 250
    openrouter_temperature: float = 0.4
    # Short on purpose: a slow primary delays the caller's reply, so the fallback should take over quickly.
    # No retries by default: every retry adds a full timeout before the fallback gets a turn.
    openrouter_timeout_ms: int = 4000
    openrouter_max_retries: int = 0

    # Fallback LLM: Groq. Unset means OpenRouter alone answers every request.
    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "openai/gpt-oss-20b"
    # Reasoning tokens count against this budget, so it must cover the thinking plus the spoken reply.
    groq_max_output_tokens: int = 1024
    groq_temperature: float = 0.4
    # Nothing follows the fallback, so it gets more time and one retry.
    groq_timeout_ms: int = 8000
    groq_max_retries: int = 1

    database_url: str
    database_pool_min_size: int = 1
    database_pool_max_size: int = 5

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

    @property
    def primary_llm_model(self) -> str:
        """Model used when an agent doesn't set llm_model: the primary provider's (OpenRouter)."""
        return self.openrouter_model

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

    @model_validator(mode="after")
    def _check_inbound_reserve(self) -> Self:
        if self.inbound_reserved_calls > self.max_concurrent_calls:
            raise ValueError("INBOUND_RESERVED_CALLS cannot exceed MAX_CONCURRENT_CALLS")
        return self

    @model_validator(mode="after")
    def _check_tts_provider(self) -> Self:
        if self.tts_provider == "bakbak" and (self.bakbak_api_key is None or not self.bakbak_voices):
            raise ValueError("TTS_PROVIDER=bakbak requires BAKBAK_API_KEY and BAKBAK_VOICES")
        if (self.tts_provider == "elevenlabs" or self.tts_fallback_enabled) and not self.elevenlabs_tts_voice_id:
            raise ValueError("ElevenLabs TTS (primary or fallback) requires ELEVENLABS_TTS_VOICE_ID")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
