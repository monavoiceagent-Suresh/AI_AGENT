"""Per-call variable-cost estimate and latency summary."""

import math
from typing import Any

from app.config import Settings


def tts_inr_per_1k_chars(settings: Settings, provider: str, model_id: str) -> float | None:
    if provider == "bakbak":
        return settings.bakbak_inr_per_1k_chars
    usd = settings.elevenlabs_tts_usd_per_1k_chars.get(model_id)
    return usd * settings.usd_to_inr if usd is not None else None


def call_cost(
    settings: Settings,
    *,
    connected_seconds: float,
    asr_seconds: float,
    tts_usage: list[tuple[int, float | None]],
    llm_cost_usd: float,
) -> dict[str, Any]:
    """`tts_usage` lists (characters, INR per 1k characters) for each TTS provider used on the call."""
    fx = settings.usd_to_inr
    asr = asr_seconds / 3600 * settings.elevenlabs_asr_usd_per_hour * fx
    unpriced = any(chars and rate is None for chars, rate in tts_usage)
    tts = None if unpriced else sum(chars / 1000 * rate for chars, rate in tts_usage if chars)
    llm = llm_cost_usd * fx
    minutes = connected_seconds / 60
    telephony = minutes * settings.exotel_inr_per_minute if settings.exotel_inr_per_minute is not None else None
    total = asr + llm + (tts or 0.0) + (telephony or 0.0)
    per_minute = total / minutes if minutes > 0 else None
    return {
        "currency": "INR",
        "asr": round(asr, 4),
        "tts": round(tts, 4) if tts is not None else None,
        "llm": round(llm, 4),
        "telephony": round(telephony, 4) if telephony is not None else None,
        "total": round(total, 4),
        "per_connected_minute": round(per_minute, 4) if per_minute is not None else None,
        "target_per_connected_minute": settings.target_cost_per_connected_minute_inr,
        "within_target": per_minute <= settings.target_cost_per_connected_minute_inr if per_minute else None,
        # Components priced as None are unknown and excluded from the total.
        "complete": tts is not None and telephony is not None,
    }


def _mean(values: list[int]) -> int | None:
    return round(sum(values) / len(values)) if values else None


def _percentile(values: list[int], pct: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(pct / 100 * len(ordered)) - 1)]


def latency_summary(turns: list[dict[str, Any]]) -> dict[str, Any]:
    def collect(key: str) -> list[int]:
        return [t[key] for t in turns if t[key] is not None]

    response = collect("response_latency_ms")
    return {
        "turns": len(turns),
        "interruptions": sum(1 for t in turns if t["interrupted"]),
        "response_ms_avg": _mean(response),
        "response_ms_p50": _percentile(response, 50),
        "response_ms_p95": _percentile(response, 95),
        "llm_ttft_ms_avg": _mean(collect("llm_ttft_ms")),
        "tts_ttfb_ms_avg": _mean(collect("tts_ttfb_ms")),
        "endpointing_ms_avg": _mean(collect("endpointing_ms")),
    }
