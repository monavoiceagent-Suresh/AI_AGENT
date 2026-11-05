-- Engine-side tenant identity, synced from Mona (PUT /tenants). Nothing here is a second source of
-- truth for tenant data: name/status mirror Mona's Client row only closely enough to let the engine
-- reject a client_id it has never heard of, or one Mona has suspended (see app/tenants.py).
--
-- No FK is added from client_id columns on agents/dispatch_jobs/inbound_routes to this table: those
-- columns were populated by flows that predate tenant validation, and a hard FK here could fail the
-- migration on a pre-existing orphaned row. Validation is enforced at the application layer instead
-- (TENANT_VALIDATION_MODE in app/config.py).
CREATE TABLE tenants (
    id        text PRIMARY KEY,
    name      text NOT NULL,
    status    text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'suspended')),
    synced_at timestamptz NOT NULL DEFAULT now()
);
