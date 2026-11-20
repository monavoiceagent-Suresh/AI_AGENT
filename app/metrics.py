"""Per-call variable-cost estimate and latency summary."""

import math
from typing import Any

from app.config import Settings


def tts_inr_per_1k_chars(settings: Settings, provider: str, model_id: str) -> float | None:
    if provider == "bakbak":
        return settings.bakbak_inr_per_1k_chars
    if provider == "sarvam":
        return settings.sarvam_inr_per_1k_chars
    usd = settings.elevenlabs_tts_usd_per_1k_chars.get(model_id)
    return usd * settings.usd_to_inr if usd is not None else None


def asr_segment_usd(settings: Settings, segment: dict[str, Any]) -> tuple[float, float] | None:
    """(base, keyterm surcharge) in USD for one ASR provider session (app/asr_recovery.Segment.record), priced
    at that provider's own rates; None for a provider with no configured rate."""
    seconds = float(segment.get("seconds") or 0.0)
    with_keyterms = bool(segment.get("keyterms_sent"))
    provider = segment.get("provider")
    if provider == "elevenlabs_scribe":
        hours = seconds / 3600
        return hours * settings.elevenlabs_asr_usd_per_hour, (
            hours * settings.elevenlabs_asr_keyterms_usd_per_hour if with_keyterms else 0.0
        )
    if provider == "deepgram_nova3":
        minutes = seconds / 60
        return minutes * settings.deepgram_asr_usd_per_minute, (
            minutes * settings.deepgram_asr_keyterms_usd_per_minute if with_keyterms else 0.0
        )
    return None


def call_cost(
    settings: Settings,
    *,
    connected_seconds: float,
    asr_seconds: float,
    tts_usage: list[tuple[int, float | None]],
    llm_cost_usd: float,
    asr_segments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """`tts_usage` lists (characters, INR per 1k characters) for each TTS provider used on the call.

    `asr_segments` (one per ASR provider session, replays included) prices ASR per provider, with the keyterm
    surcharge only for sessions that sent keyterms. Without it (callers predating per-provider tracking), ASR is
    `asr_seconds` at the ElevenLabs rate, as it always was."""
    fx = settings.usd_to_inr
    asr: float | None
    asr_keyterms = 0.0
    if asr_segments is None:
        asr = asr_seconds / 3600 * settings.elevenlabs_asr_usd_per_hour * fx
    else:
        priced = [asr_segment_usd(settings, s) for s in asr_segments]
        if any(p is None for p, s in zip(priced, asr_segments, strict=True) if s.get("seconds")):
            asr = None  # a provider without a rate: unknown, never guessed
        else:
            asr = sum((base + extra) * fx for base, extra in (p for p in priced if p is not None))
            asr_keyterms = sum(extra * fx for _, extra in (p for p in priced if p is not None))
    unpriced = any(chars and rate is None for chars, rate in tts_usage)
    tts = None if unpriced else sum(chars / 1000 * rate for chars, rate in tts_usage if chars)
    llm = llm_cost_usd * fx
    minutes = connected_seconds / 60
    telephony = minutes * settings.exotel_inr_per_minute if settings.exotel_inr_per_minute is not None else None
    total = (asr or 0.0) + llm + (tts or 0.0) + (telephony or 0.0)
    per_minute = total / minutes if minutes > 0 else None
    return {
        "currency": "INR",
        "asr": round(asr, 4) if asr is not None else None,
        # The part of "asr" that is the keyterm surcharge (already included in "asr" and "total").
        "asr_keyterms": round(asr_keyterms, 4),
        "tts": round(tts, 4) if tts is not None else None,
        "llm": round(llm, 4),
        "telephony": round(telephony, 4) if telephony is not None else None,
        "total": round(total, 4),
        "per_connected_minute": round(per_minute, 4) if per_minute is not None else None,
        "target_per_connected_minute": settings.target_cost_per_connected_minute_inr,
        "within_target": per_minute <= settings.target_cost_per_connected_minute_inr if per_minute else None,
        # Components priced as None are unknown and excluded from the total.
        "complete": asr is not None and tts is not None and telephony is not None,
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
