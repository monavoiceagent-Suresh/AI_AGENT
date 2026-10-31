import pytest

from app.calls import fallback_voice_for, voice_for
from app.config import Settings
from app.interfaces import Voice
from app.models import AgentIn
from tests.fakes import make_settings


def agent(language: str) -> AgentIn:
    return AgentIn(name="A", language=language, instructions="x", greeting="Hi")


def test_default_configuration_is_elevenlabs_premium_primary_bakbak_fallback():
    # make_settings()'s fixture base deliberately nulls out tts_fallback_provider (see tests/fakes.py) so that
    # tests picking only a primary provider never hit an accidental == collision; the *class* defaults this
    # test checks are the product's actual recommended defaults, so they are asserted against the real
    # Settings field defaults, not through that fixture override.
    assert Settings.model_fields["tts_provider"].default == "elevenlabs_premium"
    assert Settings.model_fields["tts_fallback_provider"].default == "bakbak"
    cfg = make_settings(tts_fallback_provider="bakbak")
    assert voice_for(agent("te"), cfg) == Voice(voice_id="el-voice", model_id="eleven_v3")
    assert fallback_voice_for(agent("te"), cfg) == Voice(voice_id="te-voice", model_id="standard")


def test_bakbak_primary_falls_back_to_elevenlabs_standard_for_supported_languages():
    cfg = make_settings(tts_provider="bakbak", tts_fallback_provider="elevenlabs_standard")
    assert fallback_voice_for(agent("en"), cfg) == Voice(voice_id="el-voice", model_id="eleven_flash_v2_5")
    assert fallback_voice_for(agent("te"), cfg) is None  # Flash v2.5 cannot speak Telugu


def test_elevenlabs_primary_falls_back_to_bakbak_voice_for_the_language():
    cfg = make_settings(tts_provider="elevenlabs_standard", tts_fallback_provider="bakbak")
    assert voice_for(agent("ta"), cfg) == Voice(voice_id="el-voice", model_id="eleven_flash_v2_5")
    assert fallback_voice_for(agent("ta"), cfg) == Voice(voice_id="ta-voice", model_id="standard")
    assert fallback_voice_for(agent("te"), cfg) == Voice(voice_id="te-voice", model_id="standard")


def test_elevenlabs_fallback_ignores_the_agents_elevenlabs_voice_override():
    cfg = make_settings(tts_provider="elevenlabs_standard", tts_fallback_provider="bakbak")
    override = AgentIn(
        name="A", language="en", instructions="x", greeting="Hi", voice_id="custom", tts_model_id="eleven_v3"
    )
    assert fallback_voice_for(override, cfg) == Voice(voice_id="en-voice", model_id="m1")


def test_no_fallback_when_disabled():
    assert (
        fallback_voice_for(agent("ta"), make_settings(tts_provider="elevenlabs_standard", tts_fallback_provider=None))
        is None
    )


def test_deepgram_aura_as_fallback_is_blocked_for_non_english_languages():
    cfg = make_settings(tts_provider="elevenlabs_standard", tts_fallback_provider="deepgram_aura")
    assert fallback_voice_for(agent("en"), cfg) == Voice(voice_id="aura-2-thalia-en", model_id="aura")
    assert fallback_voice_for(agent("ta"), cfg) is None
    assert fallback_voice_for(agent("hi"), cfg) is None
    assert fallback_voice_for(agent("te"), cfg) is None


def test_primary_and_fallback_provider_must_differ():
    with pytest.raises(ValueError, match="TTS_FALLBACK_PROVIDER must differ from TTS_PROVIDER"):
        make_settings(tts_provider="bakbak", tts_fallback_provider="bakbak")


def test_primary_provider_needs_its_credentials_but_fallback_does_not():
    with pytest.raises(ValueError, match="TTS_PROVIDER=elevenlabs_premium is missing its required credentials"):
        make_settings(elevenlabs_tts_voice_id="")
    with pytest.raises(ValueError, match="TTS_PROVIDER=deepgram_aura is missing its required credentials"):
        make_settings(tts_provider="deepgram_aura", deepgram_api_key=None)
    # A fallback with no credentials is accepted at settings load time; main.py simply builds no fallback for it
    # (see app/main.py build_tts_providers), the same tolerant pattern the LLM primary/fallback already uses.
    assert (
        make_settings(tts_fallback_provider="deepgram_aura", deepgram_api_key=None).tts_fallback_provider
        == "deepgram_aura"
    )
