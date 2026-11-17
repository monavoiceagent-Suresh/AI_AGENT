"""The LLM providers this engine has keys for and the models each one currently offers (GET /llm/options).

Never a hardcoded list: OpenRouter and Groq both publish an OpenAI-style `GET {base_url}/models`. Answers are
cached briefly because the OpenRouter catalogue is large and the dashboard asks every time its form opens.
"""

import time
from typing import Any

import httpx

from app.config import LLMProvider, Settings

CACHE_SECONDS = 600
TIMEOUT_SECONDS = 10
_cache: dict[str, tuple[float, list[dict[str, str]]]] = {}


def _parse(payload: Any) -> list[dict[str, str]]:
    items = payload.get("data") if isinstance(payload, dict) else None
    models = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]:
            name = item.get("name")
            models.append({"id": item["id"], "name": name if isinstance(name, str) and name else item["id"]})
    return sorted(models, key=lambda m: m["id"])


async def _fetch(http: httpx.AsyncClient, settings: Settings, provider: LLMProvider) -> list[dict[str, str]]:
    cached = _cache.get(provider)
    if cached and time.monotonic() - cached[0] < CACHE_SECONDS:
        return cached[1]
    if provider == "groq":
        base, key = settings.groq_base_url, settings.groq_api_key
    else:
        base, key = settings.openrouter_base_url, settings.openrouter_api_key
    try:
        response = await http.get(
            f"{base.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {key.get_secret_value()}"} if key else {},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        models = _parse(response.json())
    except (httpx.HTTPError, ValueError):
        # A provider's catalogue being down must not hide the others, nor stop the form from opening.
        return cached[1] if cached else []
    _cache[provider] = (time.monotonic(), models)
    return models


async def llm_options(http: httpx.AsyncClient, settings: Settings) -> dict[str, Any]:
    providers: list[LLMProvider] = [p for p in ("openrouter", "groq") if settings.llm_credentials_present(p)]
    return {
        "providers": providers,
        "defaultProvider": settings.llm_provider,
        "defaultModels": {p: settings.llm_model_for(p) for p in providers},
        "models": {p: await _fetch(http, settings, p) for p in providers},
    }
