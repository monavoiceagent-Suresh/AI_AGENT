-- An agent that belongs to a workspace can never be the engine-wide default: which persona answers a number is
-- decided by that number's published route. The flag was ignored on such agents anyway, so clear it, and stop it
-- being set again, so that "default" can never look like a way of picking a persona.
UPDATE agents SET is_inbound_default = false WHERE is_inbound_default AND client_id IS NOT NULL;

ALTER TABLE agents
    ADD CONSTRAINT agents_default_belongs_to_no_workspace CHECK (NOT is_inbound_default OR client_id IS NULL);

-- Calls that reached the engine for a number with no published route, so a wrong or missing route is visible
-- instead of silently answered by the default agent (or refused).
CREATE TABLE inbound_unrouted (
    phone            text PRIMARY KEY,
    raw_example      text NOT NULL,
    calls            integer NOT NULL DEFAULT 0,
    first_seen       timestamptz NOT NULL DEFAULT now(),
    last_seen        timestamptz NOT NULL DEFAULT now(),
    last_answered_by text NOT NULL CHECK (last_answered_by IN ('default_agent', 'refused'))
);
