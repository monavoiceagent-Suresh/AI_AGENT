"""The OpenRouter models this engine can use (GET /llm/options) and whether a primary/fallback pair is usable
(POST /llm/validate).

Never a hardcoded list: OpenRouter publishes `GET {base_url}/models`, with each model's modalities and the request
parameters it supports. Answers are cached briefly because the catalogue is large and the dashboard asks every
time its form opens. OpenRouter is the only LLM gateway, so only its catalogue is offered.
"""

import time
from typing import Any

import httpx

from app.config import Settings

CACHE_SECONDS = 600
TIMEOUT_SECONDS = 10
_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}

# What the voice pipeline sends and needs back (app/providers/chat_completions.py): streamed chat text, with a
# token budget. The client does not use tool calling, so tool support is reported but not required.
REQUIRED_PARAMETERS = ("max_tokens",)


def _compatibility(item: dict[str, Any]) -> list[str]:
    """Why this catalogue entry cannot serve a voice call (empty when it can)."""
    problems = []
    architecture = item.get("architecture") if isinstance(item.get("architecture"), dict) else {}
    inputs, outputs = architecture.get("input_modalities"), architecture.get("output_modalities")
    if isinstance(inputs, list) and "text" not in inputs:
        problems.append("does not accept text input")
    if isinstance(outputs, list) and "text" not in outputs:
        problems.append("does not produce text output")
    supported = item.get("supported_parameters")
    if isinstance(supported, list):
        missing = [p for p in REQUIRED_PARAMETERS if p not in supported]
        if missing:
            problems.append(f"does not support {', '.join(missing)}")
    return problems


def _parse(payload: Any) -> list[dict[str, Any]]:
    items = payload.get("data") if isinstance(payload, dict) else None
    models = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]:
            name = item.get("name")
            supported = item.get("supported_parameters")
            problems = _compatibility(item)
            models.append(
                {
                    "id": item["id"],
                    "name": name if isinstance(name, str) and name else item["id"],
                    "compatible": not problems,
                    "problems": problems,
                    "supportsTools": isinstance(supported, list) and "tools" in supported,
                }
            )
    return sorted(models, key=lambda m: m["id"])


async def _fetch(http: httpx.AsyncClient, settings: Settings) -> list[dict[str, Any]] | None:
    """OpenRouter's catalogue, or None when it cannot be read (and nothing is cached)."""
    cached = _cache.get("openrouter")
    if cached and time.monotonic() - cached[0] < CACHE_SECONDS:
        return cached[1]
    key = settings.openrouter_api_key
    try:
        response = await http.get(
            f"{settings.openrouter_base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {key.get_secret_value()}"} if key else {},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        models = _parse(response.json())
    except (httpx.HTTPError, ValueError):
        # The catalogue being down must not stop the form from opening.
        return cached[1] if cached else None
    _cache["openrouter"] = (time.monotonic(), models)
    return models


async def llm_options(http: httpx.AsyncClient, settings: Settings) -> dict[str, Any]:
    models = await _fetch(http, settings) or []
    return {
        "providers": ["openrouter"],
        "defaultProvider": "openrouter",
        "defaultModels": {"openrouter": settings.openrouter_model or ""},
        "defaultFallbackModel": settings.llm_fallback_model or "",
        "models": {"openrouter": [{"id": m["id"], "name": m["name"]} for m in models if m["compatible"]]},
    }


async def validate_pair(
    http: httpx.AsyncClient, settings: Settings, primary: str | None, fallback: str | None
) -> dict[str, Any]:
    """Whether a workspace's primary/fallback model choice can serve calls. None means "the engine default".
    Refuses (rather than guesses) when the catalogue cannot be read, so nothing unverified is saved."""
    effective_primary = (primary or "").strip() or settings.openrouter_model or ""
    effective_fallback = (fallback or "").strip() or settings.llm_fallback_model or ""
    problems: list[str] = []
    catalogue = await _fetch(http, settings)
    if catalogue is None:
        return {
            "ok": False,
            "primaryModel": effective_primary,
            "fallbackModel": effective_fallback,
            "problems": ["OpenRouter's model catalogue could not be read, so the models cannot be verified"],
        }
    by_id = {m["id"]: m for m in catalogue}
    for role, model in (("Primary", effective_primary), ("Fallback", effective_fallback)):
        if not model:
            if role == "Primary":
                problems.append("No primary model is configured")
            continue
        entry = by_id.get(model)
        if entry is None:
            problems.append(f"{role} model {model!r} is not offered by OpenRouter")
        elif not entry["compatible"]:
            problems.append(f"{role} model {model!r} {'; '.join(entry['problems'])}")
    if effective_fallback and effective_fallback == effective_primary:
        problems.append("The fallback model must be different from the primary model")
    return {
        "ok": not problems,
        "primaryModel": effective_primary,
        "fallbackModel": effective_fallback,
        "problems": problems,
    }
