"""LLM pricing estimates (USD per million tokens).

These values are cached snapshots and labeled as estimates wherever they are
logged. Each provider's billing dashboard is the canonical source; this table
is here only to give the user a rough per-call cost in the pipeline log.

**Model ids are matched by longest prefix**, so dated variants
(``claude-haiku-4-5-20251001``) resolve to the family rate automatically. The
sort is what makes overlapping families safe: ``claude-opus-4-5`` is tested
before ``claude-opus-4``, so the more specific row always wins regardless of
dict insertion order. Adding a row never silently shadows an existing one.

Adding a provider means adding rows here plus, if it bills server-side search
differently, a row in :data:`_SEARCH_USD_PER_CALL`.

Snapshot date: **2026-08-17**. Two rows carry expiry dates — see
:data:`PRICING_REVIEW_DATES`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# (input, output, cache_read, cache_write) in USD per 1M tokens.
#
# Cache columns follow each provider's own model:
#   - Anthropic bills cache reads at 0.1x input and 5-minute cache writes at
#     1.25x input. Both are real line items.
#   - OpenAI and Google do not charge a write premium (caching is automatic on
#     OpenAI; Google bills cache *storage* per hour, not per token), so
#     cache_write mirrors the input rate and cache_read is the discounted rate.
#     A pipeline that starts leaning on Google's explicit caching should price
#     the storage hours separately — this table cannot see them.
_PRICING_USD_PER_MTOK: dict[str, tuple[float, float, float, float]] = {
    # ---- Anthropic ----------------------------------------------------
    "claude-fable-5": (10.00, 50.00, 1.00, 12.50),
    "claude-mythos-5": (10.00, 50.00, 1.00, 12.50),
    "claude-opus-5": (5.00, 25.00, 0.50, 6.25),
    "claude-opus-4-8": (5.00, 25.00, 0.50, 6.25),
    "claude-opus-4-7": (5.00, 25.00, 0.50, 6.25),
    "claude-opus-4-6": (5.00, 25.00, 0.50, 6.25),
    # Legacy row, carried over from the April 2026 snapshot and NOT re-verified
    # in the 2026-08 pass. The whole Opus line is 5/25 today, so if anything
    # ever routes to 4.5 again, check this number before trusting the log.
    "claude-opus-4-5": (15.00, 75.00, 1.50, 18.75),
    "claude-sonnet-5": (3.00, 15.00, 0.30, 3.75),
    "claude-sonnet-4-6": (3.00, 15.00, 0.30, 3.75),
    "claude-sonnet-4-5": (3.00, 15.00, 0.30, 3.75),
    "claude-sonnet-4": (3.00, 15.00, 0.30, 3.75),
    "claude-haiku-4-5": (1.00, 5.00, 0.10, 1.25),
    # ---- Google Gemini ------------------------------------------------
    # 3.7 Flash is on introductory pricing through 2026-12-31; it doubles to
    # (1.50, 7.50) on 2027-01-01. See PRICING_REVIEW_DATES.
    "gemini-3.7-flash": (0.75, 3.75, 0.075, 0.75),
    "gemini-3.5-flash": (1.50, 9.00, 0.15, 1.50),
    "gemini-3.1-pro": (2.00, 12.00, 0.20, 2.00),
    # No ``gemini-2.5-flash-lite`` row on purpose: probed 2026-08-17 and the API
    # answers 404 "no longer available". A priced row for a retired id is worse
    # than no row — it makes a dead route look configured.
    # ---- OpenAI -------------------------------------------------------
    # ⚠️ Nothing in Prism routes to OpenAI for completions today (the key is
    # used only for the free Moderation API). These rows are staged so a future
    # route logs *something* instead of falling through to the Sonnet default —
    # but the exact API model ids were taken from pricing coverage, not from a
    # live /v1/models call. Verify the id strings before trusting the numbers.
    "gpt-5.6-sol": (5.00, 30.00, 0.50, 5.00),
    "gpt-5.6-terra": (2.00, 12.00, 0.20, 2.00),
    "gpt-5.6-luna": (0.20, 1.20, 0.02, 0.20),
    "gpt-5.5": (5.00, 30.00, 0.50, 5.00),
    "gpt-5": (1.25, 10.00, 0.125, 1.25),
    "gpt-4o": (2.50, 10.00, 0.25, 2.50),
    "default": (3.00, 15.00, 0.30, 3.75),
}

# Rows whose price is scheduled to change. Nothing reads this at runtime — it
# exists so the next person greps for "why is the log off" and finds the date
# instead of re-deriving it.
PRICING_REVIEW_DATES: dict[str, str] = {
    "gemini-3.7-flash": "2027-01-01 — introductory rate ends, doubles to (1.50, 7.50)",
    "claude-sonnet-5": "2026-08-31 — introductory (2.00, 10.00) ends; table already carries the standard rate",
}

# Server-side search, USD per request, by model prefix. Providers bill this
# per search rather than per token, and the rates are not close to each other.
#
# ⚠️ Google's rate is the MARGINAL one: the first 5,000 grounded requests per
# month are free across the Gemini 3.x family, and this table is stateless —
# it cannot know where in the month a call lands. Pricing every grounded call
# at the above-quota rate makes the log a CEILING, never an understatement.
# At Prism's ~115 web_anchor calls/month the true cost is $0.
_SEARCH_USD_PER_CALL: dict[str, float] = {
    "claude": 10.0 / 1000.0,  # Anthropic web_search tool
    "gemini": 14.0 / 1000.0,  # Google Search grounding, above the free 5k/month
}
_DEFAULT_SEARCH_USD_PER_CALL = 10.0 / 1000.0

# Open-weight model families that run locally (Ollama) — zero API cost. Ollama's
# ``family:tag`` naming (colon) is the reliable discriminator: no cloud model id
# in the pricing table uses a colon. The family list is a defensive fallback for
# untagged local names.
_LOCAL_MODEL_PREFIXES = ("qwen", "llama", "gemma", "mistral", "mixtral", "deepseek", "phi")

# Longest first, so "claude-opus-4-5" is tested before "claude-opus-4" and a
# newly added specific row can never be shadowed by a shorter family row.
_MATCH_ORDER: tuple[str, ...] = tuple(
    sorted((k for k in _PRICING_USD_PER_MTOK if k != "default"), key=len, reverse=True)
)


def is_local_model(model: str) -> bool:
    """True for models that run on the user's own hardware (Ollama) → cost $0.

    Without this, unknown models fall through to the ``default`` (Sonnet) rate,
    so local generations would be logged as if they cost money — hiding the
    whole point of routing to a local model.
    """
    if not model:
        return False
    if ":" in model:  # Ollama ``name:tag``; cloud model ids never use a colon
        return True
    lowered = model.lower()
    return any(lowered.startswith(prefix) for prefix in _LOCAL_MODEL_PREFIXES)


def pricing_per_mtok(model: str) -> tuple[float, float, float, float]:
    """Return (input, output, cache_read, cache_write) USD per 1M tokens.

    Local (Ollama) models return all-zero rates — they run for free. Otherwise
    matches by longest prefix so dated variants
    (``claude-haiku-4-5-20251001``) resolve to the family rate, falling back to
    the ``default`` row when ``model`` is unknown or empty.
    """
    if not model:
        return _PRICING_USD_PER_MTOK["default"]
    if is_local_model(model):
        return (0.0, 0.0, 0.0, 0.0)
    for key in _MATCH_ORDER:
        if model.startswith(key):
            return _PRICING_USD_PER_MTOK[key]
    return _PRICING_USD_PER_MTOK["default"]


def search_usd_per_call(model: str) -> float:
    """USD per server-side search request for ``model``'s provider.

    Local models never search through a billed endpoint, so they return 0.
    Unknown models fall back to the Anthropic rate — the conservative choice,
    since it is the higher of the two token-side defaults this table ships.
    """
    if not model or is_local_model(model):
        return 0.0
    for prefix, rate in _SEARCH_USD_PER_CALL.items():
        if model.startswith(prefix):
            return rate
    return _DEFAULT_SEARCH_USD_PER_CALL


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
        + web_searches * search_usd_per_call(model or "")
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


__all__ = [
    "PRICING_REVIEW_DATES",
    "compute_call_cost_usd",
    "is_local_model",
    "pricing_per_mtok",
    "search_usd_per_call",
    "summarize_costs",
]
