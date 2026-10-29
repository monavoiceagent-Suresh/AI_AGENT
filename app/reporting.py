"""Reports finished inbound calls to the dashboard through the durable webhook outbox.

The event is written in the same transaction that finishes the call, so a call that was saved is always
reported, and delivery happens later in the dispatcher loop, never on the call's own path.
"""

import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import asyncpg

from app.phone import normalize_phone

SCHEMA_VERSION = 1
SUMMARY_LIMIT = 4000


def inbound_call_event(
    call: Mapping[str, Any], *, has_transcript: bool, now: datetime | None = None, event_id: str | None = None
) -> dict[str, Any]:
    """The `call.completed` event for a finished inbound call.

    `transcriptRef` is the engine call id, an opaque key and not a link: the transcript is read with the
    engine's authenticated `GET /calls/{id}`, which only the dashboard's server can call.
    """
    # The number the call was routed by; the stream's own `to` only for calls recorded before that was kept.
    dialled = call.get("dialled_number") or normalize_phone(call["to_number"])
    caller = (call["from_number"] or "").strip()
    return {
        "schemaVersion": SCHEMA_VERSION,
        "eventId": event_id or str(uuid.uuid4()),
        "eventType": "call.completed",
        "occurredAt": (now or datetime.now(UTC)).isoformat(),
        "data": {
            "direction": "inbound",
            "engineCallId": str(call["id"]),
            "dialledNumber": dialled if dialled else (call["to_number"] or "").strip(),
            "callerNumber": caller or None,
            "durationSeconds": round(call["connected_seconds"] or 0),
            "status": call["status"],
            "outcome": call["outcome"],
            "summary": (call["summary"] or "")[:SUMMARY_LIMIT],
            "collectedFields": call["outcome_data"] or {},
            "transcriptRef": str(call["id"]) if has_transcript else None,
        },
    }


async def enqueue_inbound_call_report(conn: asyncpg.Connection, call_id: int, webhook_url: str) -> bool:
    """Queue the report for a finished inbound call (once per call). False for other calls."""
    call = await conn.fetchrow(
        "SELECT id, direction, from_number, to_number, dialled_number, status, "
        "connected_seconds::float8 AS connected_seconds, "
        "outcome, summary, outcome_data FROM calls WHERE id = $1",
        call_id,
    )
    if call is None or call["direction"] != "inbound":
        return False
    has_transcript = await conn.fetchval("SELECT EXISTS (SELECT 1 FROM call_turns WHERE call_id = $1)", call_id)
    event = inbound_call_event(call, has_transcript=has_transcript)
    inserted = await conn.fetchval(
        "INSERT INTO webhook_outbox (event_id, call_id, webhook_url, payload) VALUES ($1, $2, $3, $4) "
        "ON CONFLICT (call_id) WHERE call_id IS NOT NULL DO NOTHING RETURNING id",
        event["eventId"],
        call_id,
        webhook_url,
        event,
    )
    return inserted is not None
