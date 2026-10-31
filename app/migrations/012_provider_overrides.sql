-- Per-agent ASR/TTS provider overrides (Mona's Persona -> Workspace -> Global precedence resolves to this
-- before it ever reaches the engine). NULL means "use this engine's global ASR_PROVIDER/TTS_PROVIDER etc."
ALTER TABLE agents
    ADD COLUMN asr_provider text,
    ADD COLUMN asr_fallback_provider text,
    ADD COLUMN tts_provider text,
    ADD COLUMN tts_fallback_provider text;
