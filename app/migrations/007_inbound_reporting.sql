-- Finished inbound calls are reported to the dashboard through the same durable outbox as campaign events.
-- They belong to no dispatch job, so the destination URL is stored on the row and each call is reported once.
ALTER TABLE webhook_outbox
    ALTER COLUMN job_id DROP NOT NULL,
    ADD COLUMN call_id bigint REFERENCES calls (id),
    ADD COLUMN webhook_url text,
    ADD COLUMN abandoned_at timestamptz,
    ADD CONSTRAINT webhook_outbox_has_destination CHECK (job_id IS NOT NULL OR webhook_url IS NOT NULL);

CREATE UNIQUE INDEX webhook_outbox_one_report_per_call ON webhook_outbox (call_id) WHERE call_id IS NOT NULL;

CREATE INDEX webhook_outbox_call_reports_pending
    ON webhook_outbox (id) WHERE job_id IS NULL AND delivered_at IS NULL AND abandoned_at IS NULL;
