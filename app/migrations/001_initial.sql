CREATE TABLE agents (
    id                 bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name               text NOT NULL,
    language           text NOT NULL CHECK (language IN ('en', 'hi', 'ta', 'te')),
    workflow           text NOT NULL CHECK (workflow IN ('general', 'lead_qualification', 'faq', 'appointment')),
    instructions       text NOT NULL,
    greeting           text NOT NULL,
    knowledge          text NOT NULL DEFAULT '',
    collect_fields     jsonb NOT NULL DEFAULT '[]',
    voice_id           text,
    tts_model_id       text,
    llm_model          text,
    is_inbound_default boolean NOT NULL DEFAULT false,
    created_at         timestamptz NOT NULL DEFAULT now(),
    updated_at         timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX agents_single_inbound_default ON agents (is_inbound_default) WHERE is_inbound_default;

CREATE TABLE calls (
    id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    agent_id          bigint NOT NULL REFERENCES agents (id),
    direction         text NOT NULL CHECK (direction IN ('inbound', 'outbound')),
    exotel_call_sid   text UNIQUE,
    stream_sid        text,
    from_number       text,
    to_number         text,
    status            text NOT NULL,
    end_reason        text,
    created_at        timestamptz NOT NULL DEFAULT now(),
    started_at        timestamptz,
    ended_at          timestamptz,
    connected_seconds numeric(10, 2),
    usage             jsonb,
    cost              jsonb,
    latency           jsonb,
    outcome           text,
    outcome_data      jsonb,
    summary           text
);

CREATE INDEX calls_created_at ON calls (created_at DESC);

CREATE TABLE call_turns (
    call_id             bigint NOT NULL REFERENCES calls (id) ON DELETE CASCADE,
    idx                 integer NOT NULL,
    user_text           text,
    agent_text          text NOT NULL,
    interrupted         boolean NOT NULL,
    endpointing_ms      integer,
    llm_ttft_ms         integer,
    tts_ttfb_ms         integer,
    response_latency_ms integer,
    PRIMARY KEY (call_id, idx)
);
