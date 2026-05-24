"""Claude API pricing estimates (USD per million tokens).

These values are cached snapshots and labeled as estimates wherever they are
logged. The billing API is the canonical source; this table is here only to
give the user a rough per-call cost in the pipeline log.

Extending pricing for new model families (OpenAI, Gemini, etc.) means adding
entries to ``_PRICING_USD_PER_MTOK`` keyed by model prefix. The prefix match
makes dated variants (``claude-haiku-4-5-20251001``) resolve to the family
rate automatically.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# (input, output, cache_read, cache_write) in USD per 1M tokens.
# Defaults track Claude Sonnet 4 pricing (April 2026 snapshot).
_PRICING_USD_PER_MTOK: dict[str, tuple[float, float, float, float]] = {
    "claude-sonnet-4-5": (3.00, 15.00, 0.30, 3.75),
    "claude-sonnet-4": (3.00, 15.00, 0.30, 3.75),  # 4.x family share Sonnet 4 rates
    "claude-opus-4-5": (15.00, 75.00, 1.50, 18.75),
    "claude-haiku-4-5": (1.00, 5.00, 0.10, 1.25),
    "default": (3.00, 15.00, 0.30, 3.75),
}

# Web search costs $10 per 1000 searches (April 2026 snapshot).
_WEB_SEARCH_USD_PER_CALL = 10.0 / 1000.0


def pricing_per_mtok(model: str) -> tuple[float, float, float, float]:
    """Return (input, output, cache_read, cache_write) USD per 1M tokens.

    Falls back to the ``default`` row when ``model`` is unknown or empty.
    Matches by prefix so dated variants (``claude-haiku-4-5-20251001``)
    still resolve to the family rate.
    """
    if not model:
        return _PRICING_USD_PER_MTOK["default"]
    for key, rates in _PRICING_USD_PER_MTOK.items():
        if key != "default" and model.startswith(key):
            return rates
    return _PRICING_USD_PER_MTOK["default"]


def compute_call_cost_usd(model: str, usage: dict[str, Any]) -> float:
    """Cost in USD for a SINGLE LLM call.

    `usage` keys (any subset, missing → 0): input_tokens, output_tokens,
    cache_read_tokens, cache_write_tokens, web_searches. Use this at the
    point where the model + usage are both known so each call carries its
    OWN USD — multi-model pipelines stop losing money to the "last model
    wins" overwrite bug.
    """
    in_rate, out_rate, cache_r_rate, cache_w_rate = pricing_per_mtok(model or "")
    input_t = int(usage.get("input_tokens", 0) or 0)
    output_t = int(usage.get("output_tokens", 0) or 0)
    cache_r = int(usage.get("cache_read_tokens", 0) or 0)
    cache_w = int(usage.get("cache_write_tokens", 0) or 0)
    web_searches = int(usage.get("web_searches", 0) or 0)

    total = (
        input_t / 1_000_000 * in_rate
        + output_t / 1_000_000 * out_rate
        + cache_r / 1_000_000 * cache_r_rate
        + cache_w / 1_000_000 * cache_w_rate
        + web_searches * _WEB_SEARCH_USD_PER_CALL
    )
    return round(total, 6)


def summarize_costs(by_model: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a per-model totals dict into a single cost-summary shape.

    `by_model` is ``{model_string: {calls, input_tokens, output_tokens,
    cache_read_tokens, cache_write_tokens, web_searches, total_usd}}`` —
    the format that pipelines accumulate locally via
    :func:`llm_kit.telemetry.make_pipeline_usage_tracker`. The returned dict
    has the legacy single-model shape PLUS a ``by_model`` breakdown so
    callers can render both totals and per-model granularity.
    """
    total_input = total_output = total_cache_r = total_cache_w = 0
    total_web_searches = total_calls = 0
    total_usd = 0.0
    breakdown: dict[str, dict[str, Any]] = {}

    for model, totals in by_model.items():
        calls = int(totals.get("calls", 0) or 0)
        in_t = int(totals.get("input_tokens", 0) or 0)
        out_t = int(totals.get("output_tokens", 0) or 0)
        cr_t = int(totals.get("cache_read_tokens", 0) or 0)
        cw_t = int(totals.get("cache_write_tokens", 0) or 0)
        ws = int(totals.get("web_searches", 0) or 0)
        usd = float(totals.get("total_usd", 0.0) or 0.0)

        total_calls += calls
        total_input += in_t
        total_output += out_t
        total_cache_r += cr_t
        total_cache_w += cw_t
        total_web_searches += ws
        total_usd += usd

        breakdown[model or "(unknown)"] = {
            "calls": calls,
            "input_tokens": in_t,
            "output_tokens": out_t,
            "cache_read_tokens": cr_t,
            "cache_write_tokens": cw_t,
            "web_searches": ws,
            "total_usd": round(usd, 6),
        }

    dominant_model = ""
    if breakdown:
        dominant_model = max(
            breakdown.items(),
            key=lambda kv: (
                kv[1]["input_tokens"]
                + kv[1]["output_tokens"]
                + kv[1]["cache_read_tokens"]
                + kv[1]["cache_write_tokens"]
            ),
        )[0]

    return {
        "model": dominant_model,
        "calls": total_calls,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "cache_read_tokens": total_cache_r,
        "cache_write_tokens": total_cache_w,
        "web_searches": total_web_searches,
        "total_usd": round(total_usd, 6),
        "by_model": breakdown,
        "estimated_at": datetime.now(timezone.utc).isoformat(),
    }


__all__ = ["compute_call_cost_usd", "pricing_per_mtok", "summarize_costs"]
