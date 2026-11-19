-- P0 reliability (2026-11): additive only, every new column nullable so older rows and older engine
-- builds keep working.
-- The workspace's own fallback OpenRouter model (NULL = the engine's LLM_FALLBACK_MODEL).
ALTER TABLE agents ADD COLUMN llm_fallback_model text;
-- Per-turn delivery evidence, kept distinct: transmitted vs confirmed played (Exotel marks), why a turn ended
-- early or failed, and which LLM model actually answered it.
ALTER TABLE call_turns
    ADD COLUMN sentences_sent integer,
    ADD COLUMN sentences_confirmed integer,
    ADD COLUMN playback_confirmed boolean,
    ADD COLUMN interrupted_by text,
    ADD COLUMN failure_reason text,
    ADD COLUMN llm_model text;
