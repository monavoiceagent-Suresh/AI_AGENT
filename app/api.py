import asyncio
import base64
import json
import logging
import secrets
import uuid
from typing import Annotated
from urllib.parse import parse_qsl

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app import calls, db, dispatch, dispatch_db, inbound, inbound_db, voices
from app.calls import CallRejected, Services
from app.models import (
    Agent,
    AgentIn,
    Call,
    CallDetail,
    CallTurn,
    CancelOut,
    DispatchIn,
    DispatchOut,
    InboundRouteOut,
    InboundRoutePublishIn,
    JobOut,
    OutboundCallIn,
    TestCallIn,
    UnroutedOut,
    VoicesOut,
)
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
    return {"status": "ok", "active_calls": svc.active_calls}


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


# ---- inbound routing (published by the Mona dashboard) ---------------------------------------


@admin.put("/inbound/routes")
async def publish_inbound_route(body: InboundRoutePublishIn, svc: Svc) -> InboundRouteOut:
    """Publish (or replace) the persona that answers one dialled number. Versions must increase."""
    try:
        return await inbound.publish(svc, body)
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
    try:
        call_id = await calls.place_outbound_call(svc, Agent(**dict(record)), body.to, body.contact_name)
    except CallRejected as exc:
        raise HTTPException(409, str(exc)) from exc
    except ExotelError as exc:
        raise HTTPException(502, str(exc)) from exc
    return await get_call(call_id, svc)


@admin.post("/calls/test", status_code=201)
async def place_test_call(body: TestCallIn, svc: Svc) -> CallDetail:
    """Places one call right away from a persona snapshot instead of a stored agent_id: for the dashboard's
    "test outbound call" control, which has no campaign or published route to have created an agent yet."""
    persona = body.persona
    speakable = AgentIn(
        **persona.model_dump(include=set(dispatch_db.PERSONA_FIELDS)),
        voice_id=body.voice_id,
        tts_model_id=body.tts_model_id,
    )
    if problem := calls.validate_agent(speakable, svc.settings):
        raise HTTPException(422, f"Persona {persona.key!r} cannot be spoken by the voice engine: {problem}")
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


# ---- Exotel ---------------------------------------------------------------------------


@public.post("/exotel/status", include_in_schema=False)
async def exotel_status(request: Request, svc: Svc, token: str = "") -> dict:
    """StatusCallback for outbound calls; authenticated by the secret in the URL we gave Exotel."""
    expected = svc.settings.exotel_webhook_secret.get_secret_value()
    if not secrets.compare_digest(token.encode(), expected.encode()):
        raise HTTPException(403, "Invalid token")
    if request.headers.get("content-type", "").startswith("application/json"):
        payload = await request.json()
    else:
        payload = dict(parse_qsl((await request.body()).decode()))
    call_sid, status = payload.get("CallSid"), payload.get("Status")
    if not call_sid or not status:
        raise HTTPException(422, "CallSid and Status are required")
    await calls.call_status(svc, str(call_sid), str(status))
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
