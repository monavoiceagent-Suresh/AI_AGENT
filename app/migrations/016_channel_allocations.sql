-- M2 shared channel allocator: the durable record of every reservation's lifecycle, reconciled
-- against Redis (the atomic fast path, app/allocator.py) and, on restart, against Exotel's own
-- call state (app/allocator.py reconcile_on_startup). Redis can be flushed/restarted and this
-- table rebuilds it; this table can lag Redis by one write and Redis remains correct meanwhile
-- (Redis is checked first for every admission decision, this table is the audit/recovery copy).
CREATE TABLE channel_allocations (
    -- Not a uuid column: outbound ids are "out-<uuid>" and inbound ids are the Exotel CallSid
    -- itself (app/allocator.py new_allocation_id, app/calls.py _resolve_inbound).
    id           text PRIMARY KEY,
    direction    text NOT NULL CHECK (direction IN ('inbound', 'outbound')),
    state        text NOT NULL CHECK (state IN ('reserved', 'connected', 'releasing', 'released')) DEFAULT 'reserved',
    call_sid     text,
    reserved_at  timestamptz NOT NULL DEFAULT now(),
    connected_at timestamptz,
    released_at  timestamptz,
    updated_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX channel_allocations_state ON channel_allocations (state);
CREATE INDEX channel_allocations_call_sid ON channel_allocations (call_sid) WHERE call_sid IS NOT NULL;
