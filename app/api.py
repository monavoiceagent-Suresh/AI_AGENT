import asyncio
import base64
import json
import logging
import secrets
import uuid
from typing import Annotated, Literal
from urllib.parse import parse_qsl

import asyncpg
import httpx
from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.background import BackgroundTask

from app import calls, db, dispatch, dispatch_db, inbound, inbound_db, monitoring, tenants, voices
from app.calls import CallRejected, Services
from app.interfaces import TTSError
from app.models import (
    Agent,
    AgentIn,
    Call,
    CallDetail,
    CallTurn,
    CancelOut,
    CapacityOut,
    DispatchIn,
    DispatchOut,
    InboundRouteOut,
    InboundRoutePublishIn,
    JobOut,
    OutboundCallIn,
    TenantOut,
    TenantSyncIn,
    TestCallIn,
    UnroutedOut,
    VoicesOut,
)
from app.phone import normalize_phone
from app.providers.exotel import ExotelError, ExotelMediaStream
from app.redaction import redact

log = logging.getLogger(__name__)

bearer = HTTPBearer()


def services(request: Request) -> Services:
    return request.app.state.services


Svc = Annotated[Services, Depends(services)]


def require_admin(credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer)], svc: Svc) -> None:
    expected = svc.settings.app_secret_key.get_secret_value()
    if not secrets.compare_digest(credentials.credentials.encode(), expected.encode()):
        raise HTTPException(401, "Invalid token")


public = APIRouter()
admin = APIRouter(dependencies=[Depends(require_admin)])


@public.get("/health")
async def health(svc: Svc) -> dict:
    await svc.pool.fetchval("SELECT 1")
    return {"status": "ok", "active_calls": await svc.allocator.occupied(), "outbound_dispatch_version": 1}


@admin.get("/capacity")
async def capacity(svc: Svc) -> CapacityOut:
    """M3 Super Admin operational visibility: live occupancy, per-tenant breakdown, outbound
    queue depth, and stale-allocation health counters. Read-only (see app/monitoring.py)."""
    return await monitoring.capacity_snapshot(svc.pool, svc.settings, svc.allocator)


# ---- agents ---------------------------------------------------------------------------


def _checked(agent: AgentIn, svc: Services) -> AgentIn:
    if error := calls.validate_agent(agent, svc.settings):
        raise HTTPException(422, error)
    return agent


PERSONA_TAKEN = "Another agent already serves this persona_key for this client_id"


@admin.post("/agents", status_code=201)
async def create_agent(agent: AgentIn, svc: Svc) -> Agent:
    try:
        return Agent(**dict(await db.create_agent(svc.pool, _checked(agent, svc))))
    except asyncpg.UniqueViolationError as exc:
        raise HTTPException(409, PERSONA_TAKEN) from exc


@admin.get("/voices")
async def list_voices(svc: Svc) -> VoicesOut:
    """The real voices the configured TTS provider has right now, for the dashboard's voice picker."""
    try:
        return await voices.list_voices(svc.http, svc.settings)
    except voices.VoicesUnavailable as exc:
        raise HTTPException(502, str(exc)) from exc


@admin.get("/agents")
async def list_agents(svc: Svc) -> list[Agent]:
    return [Agent(**dict(r)) for r in await db.list_agents(svc.pool)]


@admin.post("/voices/preview")
async def voice_preview(svc: Svc, voice_id: Annotated[str, Query(pattern=r"^[A-Za-z0-9_-]{1,128}$")]) -> Response:
    try:
        audio = await voices.preview_voice(svc, voice_id)
    except (voices.VoicesUnavailable, TTSError, TimeoutError, httpx.HTTPError, OSError, ValueError) as exc:
        raise HTTPException(502, "Voice preview is unavailable") from exc
    if audio is None:
        raise HTTPException(404, "Voice is no longer available")
    return Response(audio, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@admin.get("/agents/{agent_id}")
async def get_agent(agent_id: int, svc: Svc) -> Agent:
    record = await db.get_agent(svc.pool, agent_id)
    if record is None:
        raise HTTPException(404, "Agent not found")
    return Agent(**dict(record))


@admin.put("/agents/{agent_id}")
async def update_agent(agent_id: int, agent: AgentIn, svc: Svc) -> Agent:
    try:
        record = await db.update_agent(svc.pool, agent_id, _checked(agent, svc))
    except asyncpg.UniqueViolationError as exc:
        raise HTTPException(409, PERSONA_TAKEN) from exc
    if record is None:
        raise HTTPException(404, "Agent not found")
    return Agent(**dict(record))


# ---- tenants (synced by the Mona dashboard; see app/tenants.py) -------------------------------


@admin.put("/tenants")
async def sync_tenant(body: TenantSyncIn, svc: Svc) -> TenantOut:
    """Idempotent upsert of one tenant's identity, so the engine can validate client_id elsewhere."""
    await tenants.upsert_tenant(svc.pool, body.id, body.name, body.status)
    return TenantOut(id=body.id, name=body.name, status=body.status)


@admin.get("/tenants/{tenant_id}")
async def get_tenant(tenant_id: str, svc: Svc) -> TenantOut:
    row = await tenants.get_tenant(svc.pool, tenant_id)
    if row is None:
        raise HTTPException(404, "Tenant not found")
    return TenantOut(id=row["id"], name=row["name"], status=row["status"])


# ---- inbound routing (published by the Mona dashboard) ---------------------------------------


@admin.put("/inbound/routes")
async def publish_inbound_route(body: InboundRoutePublishIn, svc: Svc) -> InboundRouteOut:
    """Publish (or replace) the persona that answers one dialled number. Versions must increase."""
    try:
        await tenants.require_known(svc, body.client_id, context="inbound route publish")
        return await inbound.publish(svc, body)
    except tenants.TenantNotRecognized as exc:
        raise HTTPException(404, {"code": "tenant_not_recognized", "message": str(exc)}) from exc
    except tenants.TenantSuspended as exc:
        raise HTTPException(403, {"code": "tenant_suspended", "message": str(exc)}) from exc
    except inbound_db.StaleVersion as exc:
        raise HTTPException(409, {"code": "stale_version", "currentVersion": exc.current}) from exc
    except inbound_db.VersionConflict as exc:
        raise HTTPException(409, {"code": "version_conflict"}) from exc
    except inbound_db.NumberTaken as exc:
        raise HTTPException(409, {"code": "number_taken"}) from exc
    except inbound.PersonaUnspeakable as exc:
        raise HTTPException(422, {"code": "persona_unspeakable", "message": str(exc)}) from exc


@admin.get("/inbound/unrouted")
async def list_unrouted_numbers(svc: Svc) -> list[UnroutedOut]:
    """Numbers that calls reached with no published route: the usual sign of a route keyed to the wrong number."""
    return [
        UnroutedOut(
            phone=row["phone"],
            raw_example=row["raw_example"],
            calls=row["calls"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            answered_by=row["answered_by"],
        )
        for row in await inbound_db.list_unrouted(svc.pool)
    ]


@admin.delete("/inbound/routes/{phone}")
async def revoke_inbound_route(phone: str, client_id: str, svc: Svc, minimum_version: int = 0) -> dict:
    from app.phone import normalize_phone

    normalized = normalize_phone(phone)
    if not normalized:
        raise HTTPException(422, "Invalid phone")
    async with svc.pool.acquire() as conn, conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1))", normalized)
        await conn.execute(
            "UPDATE inbound_routes SET active = false, version = GREATEST(version + 1, $3) "
            "WHERE phone = $1 AND client_id = $2",
            normalized,
            client_id,
            minimum_version,
        )
        # Caller IDs are job snapshots. Revocation must also stop future calls using that snapshot.
        await conn.execute(
            "UPDATE dispatch_jobs SET status = 'cancelled' WHERE client_id = $1 "
            "AND caller_id = $2 AND status IN ('queued', 'running')",
            client_id,
            f"+{normalized}",
        )
    return {"ok": True}


@admin.get("/inbound/routes/{phone}")
async def get_inbound_route(phone: str, svc: Svc) -> InboundRouteOut:
    route = await inbound.get_route(svc, phone)
    if route is None:
        raise HTTPException(404, "No route is published for this number")
    return route


# ---- campaign dispatch (called by the Mona dashboard) ----------------------------------------


@admin.post("/campaigns/dispatch")
async def dispatch_campaign(body: DispatchIn, svc: Svc) -> DispatchOut:
    return await dispatch.accept(svc, body)


@admin.get("/jobs/{job_id}")
async def get_job(job_id: uuid.UUID, svc: Svc) -> JobOut:
    job = await dispatch_db.get_job(svc.pool, str(job_id))
    if job is None:
        raise HTTPException(404, "Job not found")
    return JobOut(
        engine_job_id=str(job["id"]),
        status=job["status"],
        calls_completed=job["calls_completed"],
        calls_failed=job["calls_failed"],
    )


@admin.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: uuid.UUID, svc: Svc) -> CancelOut:
    """Stops new calls. A call already in progress finishes, then the dashboard gets job.cancelled."""
    status = await dispatch_db.cancel_job(svc.pool, str(job_id))
    if status is None:
        raise HTTPException(404, "Job not found")
    return CancelOut(status=status)


# ---- calls ----------------------------------------------------------------------------


@admin.post("/calls", status_code=201)
async def place_call(body: OutboundCallIn, svc: Svc) -> CallDetail:
    record = await db.get_agent(svc.pool, body.agent_id)
    if record is None:
        raise HTTPException(404, "Agent not found")
    agent = Agent(**dict(record))
    try:
        await tenants.require_known(svc, agent.client_id, context="ad-hoc outbound call")
        call_id = await calls.place_outbound_call(svc, agent, body.to, body.contact_name)
    except tenants.TenantNotRecognized as exc:
        raise HTTPException(404, {"code": "tenant_not_recognized", "message": str(exc)}) from exc
    except tenants.TenantSuspended as exc:
        raise HTTPException(403, {"code": "tenant_suspended", "message": str(exc)}) from exc
    except CallRejected as exc:
        raise HTTPException(409, str(exc)) from exc
    except ExotelError as exc:
        raise HTTPException(502, str(exc)) from exc
    return await get_call(call_id, svc)


@admin.post("/calls/test", status_code=201)
async def place_test_call(body: TestCallIn, svc: Svc) -> CallDetail | DispatchOut:
    """Places one call right away from a persona snapshot instead of a stored agent_id: for the dashboard's
    "test outbound call" control, which has no campaign or published route to have created an agent yet."""
    if body.idempotency_key:
        # One lead, one attempt, no campaign mutation: the same queue supplies idempotency,
        # pinned persona/voice, failure reporting and durable callbacks for a test call.
        return await dispatch.accept(
            svc,
            DispatchIn(
                idempotency_key=body.idempotency_key,
                campaign_id=f"test-{body.idempotency_key}",
                client_id=body.client_id,
                persona=body.persona,
                calling_window={"start": "00:00", "end": "23:59"},
                retry_policy={"max_tries": 1, "retry_gap_minutes": 1, "callback_window_days": 1},
                leads=[{"id": body.idempotency_key, "phone": body.to, "name": body.contact_name}],
                webhook_url=body.webhook_url,
                voice_id=body.voice_id,
                tts_model_id=body.tts_model_id,
            ),
        )
    persona = body.persona
    speakable = AgentIn(
        **persona.model_dump(include=set(dispatch_db.PERSONA_FIELDS)),
        voice_id=body.voice_id,
        tts_model_id=body.tts_model_id,
    )
    if problem := calls.validate_agent(speakable, svc.settings):
        raise HTTPException(422, f"Persona {persona.key!r} cannot be spoken by the voice engine: {problem}")
    try:
        await tenants.require_known(svc, body.client_id, context="ad-hoc test call")
    except tenants.TenantNotRecognized as exc:
        raise HTTPException(404, {"code": "tenant_not_recognized", "message": str(exc)}) from exc
    except tenants.TenantSuspended as exc:
        raise HTTPException(403, {"code": "tenant_suspended", "message": str(exc)}) from exc
    async with svc.pool.acquire() as conn:
        agent_id = await dispatch_db.upsert_persona_agent(conn, body.client_id, persona)
    agent = Agent(**dict(await db.get_agent(svc.pool, agent_id)))
    try:
        call_id = await calls.place_outbound_call(
            svc, agent, body.to, body.contact_name, voice_id=body.voice_id, tts_model_id=body.tts_model_id
        )
    except CallRejected as exc:
        raise HTTPException(409, str(exc)) from exc
    except ExotelError as exc:
        raise HTTPException(502, str(exc)) from exc
    return await get_call(call_id, svc)


@admin.get("/calls")
async def list_calls(svc: Svc, limit: Annotated[int, Query(ge=1, le=500)] = 50) -> list[Call]:
    return [Call(**dict(r)) for r in await db.list_calls(svc.pool, limit)]


@admin.get("/calls/{call_id}")
async def get_call(call_id: int, svc: Svc) -> CallDetail:
    found = await db.get_call(svc.pool, call_id)
    if found is None:
        raise HTTPException(404, "Call not found")
    call, turns = found
    return CallDetail(**dict(call), turns=[CallTurn(**dict(t)) for t in turns])


@admin.post("/calls/{call_id}/recording")
async def call_recording(
    call_id: int,
    svc: Svc,
    client_id: Annotated[str, Body()],
    direction: Annotated[Literal["inbound", "outbound"], Body()],
    dialled_number: Annotated[str, Body()],
) -> StreamingResponse:
    row = await svc.pool.fetchrow(
        "SELECT c.exotel_call_sid, c.recording_url, c.direction, c.to_number, c.dialled_number, a.client_id "
        "FROM calls c JOIN agents a ON a.id = c.agent_id WHERE c.id = $1",
        call_id,
    )
    expected_number = normalize_phone(dialled_number)
    if (
        not row
        or not row["exotel_call_sid"]
        or row["client_id"] != client_id
        or row["direction"] != direction
        or not expected_number
        or expected_number != normalize_phone(row["dialled_number"] or row["to_number"])
    ):
        raise HTTPException(404, "Recording unavailable")

    sid = row["exotel_call_sid"]
    stored_url = row["recording_url"]
    try:
        audio = await svc.exotel.open_recording(stored_url) if stored_url else None
        if audio is None:
            stable_url, signed_url = await svc.exotel.recording_urls(sid)
            if stable_url and stable_url != stored_url and len(stable_url) <= 4096:
                await db.save_recording_url(svc.pool, sid, stable_url)
            if signed_url:
                audio = await svc.exotel.open_recording(signed_url)
            if audio is None and stable_url and stable_url != stored_url:
                audio = await svc.exotel.open_recording(stable_url)
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        # Provider errors can include signed URLs. Return a generic response without logging them.
        raise HTTPException(502, "Recording could not be loaded") from None
    if audio is None:
        raise HTTPException(404, "Recording unavailable")
    content_type = audio.headers.get("content-type", "").split(";", 1)[0].lower()
    return StreamingResponse(
        audio.aiter_bytes(),
        media_type=content_type if content_type.startswith("audio/") else "audio/mpeg",
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
        background=BackgroundTask(audio.aclose),
    )


# ---- Exotel ---------------------------------------------------------------------------


@public.post("/exotel/status", include_in_schema=False)
async def exotel_status(request: Request, svc: Svc, token: str = "") -> dict:
    """StatusCallback for outbound calls; authenticated by the secret in the URL we gave Exotel."""
    expected = svc.settings.exotel_webhook_secret.get_secret_value()
    if not secrets.compare_digest(token.encode(), expected.encode()):
        raise HTTPException(403, "Invalid token")
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type == "application/json":
        payload = await request.json()
        if not isinstance(payload, dict):
            raise HTTPException(422, "CallSid and Status are required")
    elif content_type == "multipart/form-data":
        async with request.form() as form:
            payload = dict(form)
    else:
        payload = dict(parse_qsl((await request.body()).decode()))
    call_sid, status = payload.get("CallSid"), payload.get("Status")
    if not isinstance(call_sid, str) or not call_sid.strip() or not isinstance(status, str) or not status.strip():
        raise HTTPException(422, "CallSid and Status are required")
    recording_url = payload.get("RecordingUrl")
    await calls.call_status(
        svc,
        call_sid.strip(),
        status.strip().lower(),
        recording_url.strip() if isinstance(recording_url, str) and len(recording_url) <= 4096 else None,
    )
    return {"ok": True}


def _exotel_authorized(ws: WebSocket, svc: Services) -> bool:
    """The Voicebot applet URL carries API_KEY:API_TOKEN, sent by Exotel as HTTP Basic auth."""
    scheme, _, encoded = ws.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "basic":
        return False
    try:
        supplied = base64.b64decode(encoded, validate=True)
    except ValueError:
        return False
    settings = svc.settings
    expected = f"{settings.exotel_api_key.get_secret_value()}:{settings.exotel_api_token.get_secret_value()}"
    return secrets.compare_digest(supplied, expected.encode())


def stream_request_summary(ws: WebSocket) -> dict:
    """The upgrade request as it is safe to log: what was asked for and which headers arrived, never their values."""
    return {
        "path": ws.url.path,
        "query": redact(dict(ws.query_params)),
        "authorization": "present" if ws.headers.get("authorization") else "absent",
        "user_agent": ws.headers.get("user-agent", "")[:60],
        "header_names": sorted(ws.headers.keys()),
    }


@public.websocket("/exotel/stream")
async def exotel_stream(ws: WebSocket) -> None:
    svc: Services = ws.app.state.services
    if not _exotel_authorized(ws, svc):
        log.warning("rejected media stream: missing or invalid Basic auth")
        await ws.close(code=1008)
        return
    await ws.accept()
    log.info("exotel stream request %s", json.dumps(stream_request_summary(ws), sort_keys=True))
    media = ExotelMediaStream(ws, queue_max_ms=svc.settings.audio_queue_max_ms)
    try:
        async with asyncio.timeout(10):
            start = await media.wait_start()
        resolved = await calls.resolve_agent(svc, start)
    except (TimeoutError, WebSocketDisconnect, CallRejected) as exc:
        log.warning("media stream not started: %s", exc or type(exc).__name__)
        await media.close()
        return
    await calls.run_call(
        svc,
        media,
        resolved.start,
        resolved.agent,
        resolved.call_id,
        resolved.caller_context,
        campaign_id=resolved.campaign_id,
    )
