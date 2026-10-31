-- The voice a call is placed with: a campaign's choice (copied from its job) or a test call's. NULL keeps the agent's own.
ALTER TABLE calls ADD COLUMN voice_id text, ADD COLUMN tts_model_id text;
ALTER TABLE dispatch_jobs ADD COLUMN voice_id text, ADD COLUMN tts_model_id text;
