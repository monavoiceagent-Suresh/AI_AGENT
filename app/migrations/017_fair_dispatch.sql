-- Persistent tenant service history survives zero-load ties and process restarts.
ALTER TABLE dispatch_jobs ADD COLUMN last_served_at timestamptz;
CREATE INDEX dispatch_jobs_tenant_service ON dispatch_jobs (client_id, last_served_at);
ALTER TABLE channel_allocations ADD COLUMN client_id text;
