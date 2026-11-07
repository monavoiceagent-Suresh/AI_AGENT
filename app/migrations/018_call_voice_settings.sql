-- A campaign's (or test call's) own tone for its chosen voice: ElevenLabs stability/style/similarity_boost.
-- NULL keeps the agent's own/default tone, same convention as voice_id/tts_model_id in 011_call_voice.sql.
ALTER TABLE calls ADD COLUMN voice_settings jsonb;
ALTER TABLE dispatch_jobs ADD COLUMN voice_settings jsonb;
