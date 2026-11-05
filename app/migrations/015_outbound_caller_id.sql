-- The outbound caller ID a dispatch job was accepted with. Nullable and snapshot-only: once a job
-- exists, this is the number its calls present as for the job's whole lifetime, even if Mona's own
-- number assignment changes afterwards (Mona resolves the value once, at dispatch time, and sends
-- it here; the engine never re-resolves it). NULL means a job created before this column existed,
-- or dispatched by a caller that did not yet resolve a tenant caller ID: those calls keep using the
-- engine-wide EXOTEL_CALLER_ID fallback exactly as every job did before this column existed.
ALTER TABLE dispatch_jobs ADD COLUMN caller_id text;
