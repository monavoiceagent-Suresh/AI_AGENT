-- Inbound routing: which published persona answers which dialled number.
-- Rows are written only by the dashboard's publish (PUT /inbound/routes); calls read them and never reach
-- back to the dashboard. The phone is normalised digits (see app/inbound.py), so the Exotel `to` value and
-- the number published from the dashboard meet in one format.
CREATE TABLE inbound_routes (
    phone        text PRIMARY KEY,
    client_id    text NOT NULL,
    agent_id     bigint NOT NULL REFERENCES agents (id),
    version      bigint NOT NULL CHECK (version >= 1),
    active       boolean NOT NULL,
    persona      jsonb NOT NULL,
    published_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX inbound_routes_client ON inbound_routes (client_id);
