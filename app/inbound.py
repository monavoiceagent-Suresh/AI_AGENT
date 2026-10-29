"""Inbound routing: the dashboard publishes number -> persona routes; calls look them up locally.

Nothing here contacts the dashboard. A call to a number with no published route is never answered with a
workspace's persona: it is refused, or (only while INBOUND_LEGACY_DEFAULT_ENABLED) answered by the
engine-wide default agent that belongs to no workspace.
"""

from typing import Any

from app import calls, inbound_db
from app.calls import Services
from app.models import AgentIn, InboundRouteOut, InboundRoutePublishIn
from app.phone import normalize_phone


class PersonaUnspeakable(Exception):
    pass


def _out(
    phone: str, status: str, *, client_id: str, version: int, active: bool, persona: dict[str, Any]
) -> InboundRouteOut:
    return InboundRouteOut(
        phone=phone,
        client_id=client_id,
        version=version,
        active=active,
        persona_key=persona["key"],
        persona_version=persona["version"],
        workflow=persona["workflow"],
        status=status,  # type: ignore[arg-type]
    )


async def publish(svc: Services, request: InboundRoutePublishIn) -> InboundRouteOut:
    phone = normalize_phone(request.phone)
    if phone is None:  # the model already enforces E.164; this keeps one definition of "valid"
        raise PersonaUnspeakable(f"{request.phone!r} is not a dialable number")
    persona = request.persona
    speakable = AgentIn(
        name=persona.name,
        language=persona.language,
        workflow=persona.workflow,
        instructions=persona.instructions,
        greeting=persona.greeting,
        knowledge=persona.knowledge,
    )
    if problem := calls.validate_agent(speakable, svc.settings):
        raise PersonaUnspeakable(f"Persona {persona.key!r} cannot be spoken by the voice engine: {problem}")
    status = await inbound_db.publish_route(svc.pool, phone, request)
    return _out(
        phone,
        status,
        client_id=request.client_id,
        version=request.version,
        active=request.active,
        persona=persona.model_dump(),
    )


async def get_route(svc: Services, raw_phone: str) -> InboundRouteOut | None:
    phone = normalize_phone(raw_phone)
    route = await inbound_db.find_route(svc.pool, phone) if phone else None
    if route is None:
        return None
    return _out(
        route["phone"],
        "current",
        client_id=route["client_id"],
        version=route["version"],
        active=route["active"],
        persona=route["persona"],
    )
