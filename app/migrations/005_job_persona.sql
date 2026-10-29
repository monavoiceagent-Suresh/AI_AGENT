-- The persona a dashboard campaign was started with. Calls of the job speak this copy, so editing the
-- persona (or starting another campaign with a newer version) never changes a campaign already running.
ALTER TABLE dispatch_jobs ADD COLUMN persona jsonb;
