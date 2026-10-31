"""Per-agent ASR/TTS provider overrides (Mona's Persona -> Workspace -> Global precedence resolves to this
before it reaches the engine): voice/validation honour the override, and resolve_tts/resolve_asr fall back to
the global default, with a warning, when the override names a provider this deployment has no credentials for."""

import pytest

from app.calls import fallback_voice_for, resolve_asr, resolve_tts, validate_agent, voice_for
from app.models import AgentIn, PersonaIn
from tests.fakes import make_settings


def agent_in(**overrides) -> AgentIn:
    base = dict(name="A", language="ta", workflow="general", instructions="i", greeting="g")
    return AgentIn(**(base | overrides))


def persona_in(**overrides) -> PersonaIn:
    base = dict(key="test-persona", name="A", language="ta", instructions="i", greeting="g", version=1)
    return PersonaIn(**(base | overrides))


class Services:
    """A bare stand-in: resolve_tts/resolve_asr only read settings, asr/tts, fallback_asr/fallback_tts and
    the two provider maps off it."""

    def __init__(self, settings, *, tts_by_provider=None, asr_by_provider=None):
        self.settings = settings
        self.tts = object()
        self.fallback_tts = object()
        self.tts_by_provider = tts_by_provider or {}
        self.asr = object()
        self.fallback_asr = object()
        self.asr_by_provider = asr_by_provider or {}


def test_agent_override_picks_the_overridden_voice_and_language_rule():
    cfg = make_settings()  # default primary elevenlabs_premium, no fallback (fixture base)
    agent = agent_in(language="ta", tts_provider="bakbak")

    assert voice_for(agent, cfg) == cfg.bakbak_voices["ta"]
    assert validate_agent(agent, cfg) is None


def test_agent_override_to_a_language_incompatible_provider_is_rejected():
    cfg = make_settings()
    agent = agent_in(language="te", tts_provider="deepgram_aura")

    assert "cannot speak language 'te'" in validate_agent(agent, cfg)


def test_agent_fallback_override_is_used_when_set():
    cfg = make_settings(tts_provider="elevenlabs_standard")
    agent = agent_in(language="ta", tts_fallback_provider="bakbak")

    assert fallback_voice_for(agent, cfg) == cfg.bakbak_voices["ta"]


def test_no_override_means_the_global_default_unchanged():
    cfg = make_settings(tts_provider="elevenlabs_standard", tts_fallback_provider="bakbak")
    agent = agent_in(language="en")

    assert voice_for(agent, cfg).model_id == "eleven_flash_v2_5"
    assert fallback_voice_for(agent, cfg) == cfg.bakbak_voices["en"]


def test_persona_in_also_validates_its_own_provider_pair():
    with pytest.raises(ValueError, match="ttsFallbackProvider must differ from ttsProvider"):
        persona_in(tts_provider="bakbak", tts_fallback_provider="bakbak")
    with pytest.raises(ValueError, match="asrFallbackProvider must differ from asrProvider"):
        persona_in(asr_provider="deepgram_nova3", asr_fallback_provider="deepgram_nova3")
    assert persona_in(tts_provider="bakbak", tts_fallback_provider="elevenlabs_standard").tts_provider == "bakbak"


def test_agent_in_rejects_the_same_primary_and_fallback_provider():
    with pytest.raises(ValueError, match="tts_fallback_provider must differ from tts_provider"):
        agent_in(tts_provider="bakbak", tts_fallback_provider="bakbak")


# ---- resolve_tts / resolve_asr -------------------------------------------------------------------


def test_resolve_tts_with_no_override_returns_the_global_instances_unchanged():
    cfg = make_settings(tts_provider="elevenlabs_standard", tts_fallback_provider="bakbak")
    svc = Services(cfg)
    agent = agent_in(language="en")

    tts, name, fallback, fallback_name = resolve_tts(svc, agent)

    assert tts is svc.tts and name == "elevenlabs_standard"
    assert fallback is svc.fallback_tts and fallback_name == "bakbak"


def test_resolve_tts_override_with_credentials_uses_the_named_providers_own_instance():
    cfg = make_settings(tts_provider="elevenlabs_standard")
    deepgram_instance = object()
    svc = Services(cfg, tts_by_provider={"deepgram_aura": deepgram_instance})
    agent = agent_in(language="en", tts_provider="deepgram_aura")

    tts, name, fallback, fallback_name = resolve_tts(svc, agent)

    assert tts is deepgram_instance and name == "deepgram_aura"
    assert fallback is None and fallback_name is None


def test_resolve_tts_override_with_no_credentials_falls_back_to_the_global_default():
    cfg = make_settings(tts_provider="elevenlabs_standard")
    svc = Services(cfg, tts_by_provider={"deepgram_aura": None})  # built, but no DEEPGRAM_API_KEY
    agent = agent_in(language="en", tts_provider="deepgram_aura")

    tts, name, fallback, fallback_name = resolve_tts(svc, agent)

    assert tts is svc.tts and name == "elevenlabs_standard"  # degraded to the global default, not a crash


def test_resolve_tts_fallback_override_with_no_credentials_disables_the_fallback_not_the_call():
    cfg = make_settings(tts_provider="elevenlabs_standard")
    svc = Services(cfg, tts_by_provider={"bakbak": None})
    agent = agent_in(language="en", tts_fallback_provider="bakbak")

    tts, name, fallback, fallback_name = resolve_tts(svc, agent)

    assert tts is svc.tts and fallback is None and fallback_name is None


def test_resolve_asr_with_no_override_returns_the_global_instances_unchanged():
    cfg = make_settings(asr_provider="elevenlabs_scribe", asr_fallback_provider="deepgram_nova3")
    svc = Services(cfg, asr_by_provider={"deepgram_nova3": object()})
    agent = agent_in(language="ta")  # no override

    asr, name, fallback, fallback_name = resolve_asr(svc, agent)

    assert asr is svc.asr and name == "elevenlabs_scribe"
    assert fallback is svc.fallback_asr and fallback_name == "deepgram_nova3"


def test_resolve_asr_override_with_credentials_uses_the_named_providers_own_instance():
    cfg = make_settings(asr_provider="elevenlabs_scribe")
    nova3 = object()
    svc = Services(cfg, asr_by_provider={"deepgram_nova3": nova3})
    agent = agent_in(language="ta", asr_provider="deepgram_nova3")

    asr, name, fallback, fallback_name = resolve_asr(svc, agent)

    assert asr is nova3 and name == "deepgram_nova3"
