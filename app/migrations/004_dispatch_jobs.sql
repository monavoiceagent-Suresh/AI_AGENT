-- Dashboard campaign dispatch: persona mapping, jobs, per-lead state and the webhook outbox.

-- An agent can serve one dashboard persona of one client. The pair is how a dispatch finds its agent.
ALTER TABLE agents
    ADD COLUMN client_id text,
    ADD COLUMN persona_key text,
    ADD CONSTRAINT agents_persona_needs_client CHECK (persona_key IS NULL OR client_id IS NOT NULL);

CREATE UNIQUE INDEX agents_client_persona ON agents (client_id, persona_key) WHERE persona_key IS NOT NULL;

CREATE TABLE dispatch_jobs (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key   text NOT NULL UNIQUE,
    client_id         text NOT NULL,
    campaign_id       text NOT NULL,
    agent_id          bigint NOT NULL REFERENCES agents (id),
    status            text NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'cancelled')),
    window_start      text NOT NULL,
    window_end        text NOT NULL,
    max_tries         integer NOT NULL CHECK (max_tries >= 1),
    retry_gap_minutes integer NOT NULL CHECK (retry_gap_minutes >= 1),
    webhook_url       text NOT NULL,
    created_at        timestamptz NOT NULL DEFAULT now(),
    -- Set when the final webhook (job.completed / job.cancelled) is queued. A campaign can have
    -- only one job that is not finalized, so a cancelled job must finish its in-flight call first.
    finalized_at      timestamptz
);

CREATE UNIQUE INDEX dispatch_jobs_one_open_per_campaign
    ON dispatch_jobs (client_id, campaign_id) WHERE finalized_at IS NULL;

CREATE TABLE dispatch_leads (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id          uuid NOT NULL REFERENCES dispatch_jobs (id) ON DELETE CASCADE,
    lead_id         text NOT NULL,
    phone           text NOT NULL,
    name            text,
    status          text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'calling', 'done', 'failed')),
    attempts        integer NOT NULL DEFAULT 0,
    next_attempt_at timestamptz NOT NULL DEFAULT now(),
    claimed_at      timestamptz,
    last_outcome    text,
    UNIQUE (job_id, lead_id)
);

CREATE INDEX dispatch_leads_due ON dispatch_leads (next_attempt_at) WHERE status = 'pending';

ALTER TABLE calls ADD COLUMN dispatch_lead_id bigint REFERENCES dispatch_leads (id);
CREATE INDEX calls_dispatch_lead ON calls (dispatch_lead_id) WHERE dispatch_lead_id IS NOT NULL;

-- Events for the dashboard. Written in the same transaction as the state change they describe,
-- delivered later and in order per job.
CREATE TABLE webhook_outbox (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_id        uuid NOT NULL UNIQUE,
    job_id          uuid NOT NULL REFERENCES dispatch_jobs (id) ON DELETE CASCADE,
    payload         jsonb NOT NULL,
    attempts        integer NOT NULL DEFAULT 0,
    next_attempt_at timestamptz NOT NULL DEFAULT now(),
    delivered_at    timestamptz,
    last_error      text,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX webhook_outbox_undelivered ON webhook_outbox (job_id, id) WHERE delivered_at IS NULL;
