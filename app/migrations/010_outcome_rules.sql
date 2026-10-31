-- A persona's own custom outcome vocabulary, from the dashboard's Call Outcome Rules page. NULL/empty
-- outcome_rules means "use the fixed per-workflow outcome list" (OUTCOMES in prompts.py), unchanged from
-- before this migration. Refreshed on every dispatch/publish/test call, exactly like instructions/knowledge.
ALTER TABLE agents ADD COLUMN outcome_rules jsonb NOT NULL DEFAULT '[]';
ALTER TABLE agents ADD COLUMN outcome_fallback text;
