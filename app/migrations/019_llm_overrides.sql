-- A workspace's (Super Admin) or persona's own LLM choice, same convention as 012_provider_overrides.sql:
-- NULL keeps the engine's global default (LLM_PROVIDER / its fallback). agents.llm_model already exists.
ALTER TABLE agents
    ADD COLUMN llm_provider text,
    ADD COLUMN llm_fallback_provider text;
