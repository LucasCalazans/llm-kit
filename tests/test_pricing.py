"""Pricing math + family-prefix resolution."""

from __future__ import annotations

import pytest

from llm_kit.pricing import (
    compute_call_cost_usd,
    is_local_model,
    pricing_per_mtok,
    search_usd_per_call,
    summarize_costs,
)


def test_pricing_per_mtok_resolves_known_family():
    rates = pricing_per_mtok("claude-sonnet-4-5")
    assert rates == (3.00, 15.00, 0.30, 3.75)


def test_pricing_per_mtok_matches_dated_variant_by_prefix():
    rates = pricing_per_mtok("claude-haiku-4-5-20251001")
    assert rates == (1.00, 5.00, 0.10, 1.25)


def test_pricing_per_mtok_falls_back_to_default_when_unknown():
    rates_unknown = pricing_per_mtok("gpt-7-omega")
    rates_default = pricing_per_mtok("default")
    assert rates_unknown == rates_default


def test_pricing_per_mtok_empty_model_falls_back():
    assert pricing_per_mtok("") == pricing_per_mtok("default")


def test_pricing_per_mtok_local_ollama_model_is_free():
    # Local models run on the user's hardware — no API cost. Without this they
    # would fall through to the Sonnet ``default`` rate.
    assert pricing_per_mtok("qwen3:30b-a3b-instruct-2507-q4_K_M") == (0.0, 0.0, 0.0, 0.0)


def test_is_local_model_discriminates_local_from_cloud():
    assert is_local_model("qwen3:30b-a3b-instruct-2507-q4_K_M")  # Ollama name:tag
    assert is_local_model("llama3.1")  # untagged local family
    assert not is_local_model("claude-sonnet-4-5")
    assert not is_local_model("gemini-2.0-flash")
    assert not is_local_model("")


def test_compute_call_cost_local_model_is_zero():
    cost = compute_call_cost_usd(
        "qwen3:30b-a3b-instruct-2507-q4_K_M",
        {"input_tokens": 1_607, "output_tokens": 287},
    )
    assert cost == 0.0


def test_compute_call_cost_sonnet_basic():
    cost = compute_call_cost_usd(
        "claude-sonnet-4-5",
        {"input_tokens": 1_000_000, "output_tokens": 0},
    )
    # 1M input tokens at $3.00 = $3.00
    assert cost == 3.0


def test_compute_call_cost_includes_cache_and_web_search():
    cost = compute_call_cost_usd(
        "claude-opus-4-5",
        {
            "input_tokens": 100_000,
            "output_tokens": 50_000,
            "cache_read_tokens": 200_000,
            "cache_write_tokens": 10_000,
            "web_searches": 5,
        },
    )
    # 100k * 15 / 1M    = 1.5
    # 50k  * 75 / 1M    = 3.75
    # 200k * 1.5 / 1M   = 0.3
    # 10k  * 18.75 / 1M = 0.1875
    # 5    * 0.01       = 0.05
    expected = round(1.5 + 3.75 + 0.3 + 0.1875 + 0.05, 6)
    assert cost == expected


def test_compute_call_cost_missing_fields_default_to_zero():
    # Only output tokens — every other field defaults to 0.
    cost = compute_call_cost_usd("claude-haiku-4-5", {"output_tokens": 200_000})
    # 200k * 5 / 1M = 1.0
    assert cost == 1.0


def test_summarize_costs_aggregates_and_picks_dominant_model():
    by_model = {
        "claude-haiku-4-5": {
            "calls": 1,
            "input_tokens": 100,
            "output_tokens": 50,
            "total_usd": 0.001,
        },
        "claude-sonnet-4-5": {
            "calls": 2,
            "input_tokens": 10_000,
            "output_tokens": 5_000,
            "total_usd": 0.05,
        },
    }
    summary = summarize_costs(by_model)
    assert summary["calls"] == 3
    assert summary["input_tokens"] == 10_100
    assert summary["output_tokens"] == 5_050
    assert summary["total_usd"] == round(0.051, 6)
    # Sonnet has more total tokens, so it should be the dominant model.
    assert summary["model"] == "claude-sonnet-4-5"
    assert set(summary["by_model"].keys()) == {"claude-haiku-4-5", "claude-sonnet-4-5"}
    assert "estimated_at" in summary


def test_summarize_costs_empty_returns_zero_totals():
    summary = summarize_costs({})
    assert summary["calls"] == 0
    assert summary["total_usd"] == 0.0
    assert summary["model"] == ""
    assert summary["by_model"] == {}


# --- longest-prefix resolution -----------------------------------------------
#
# The regression these guard: before 2026-08 the table matched in dict-insertion
# order, so a model id could resolve to a SHORTER family row that happened to be
# declared first, and any id newer than the table (``claude-opus-4-6``) fell all
# the way through to the Sonnet ``default`` — pricing Opus calls ~40% under.


@pytest.mark.parametrize(
    ("model", "expected_input_rate"),
    [
        ("claude-opus-4-5", 15.00),  # specific row wins over "claude-opus-4-…"
        ("claude-opus-4-6", 5.00),  # current Opus line, was hitting default
        ("claude-opus-5", 5.00),
        ("claude-sonnet-4-6", 3.00),
        ("claude-sonnet-5", 3.00),
        ("claude-fable-5", 10.00),
        ("gemini-3.7-flash", 0.75),
        ("gpt-5.6-luna", 0.20),
        ("gpt-5.6-sol", 5.00),
        ("gpt-5", 1.25),  # shorter row still reachable when nothing longer matches
    ],
)
def test_pricing_per_mtok_prefers_longest_matching_prefix(model, expected_input_rate):
    assert pricing_per_mtok(model)[0] == expected_input_rate


def test_pricing_per_mtok_opus_no_longer_falls_through_to_default():
    assert pricing_per_mtok("claude-opus-4-8") != pricing_per_mtok("default")


# --- provider-aware search pricing -------------------------------------------


def test_search_rate_differs_by_provider():
    assert search_usd_per_call("claude-sonnet-4-6") == 0.010
    assert search_usd_per_call("gemini-3.7-flash") == 0.014


def test_search_rate_is_zero_for_local_models():
    assert search_usd_per_call("qwen3:30b-a3b-instruct-2507-q4_K_M") == 0.0
    assert search_usd_per_call("") == 0.0


def test_search_rate_unknown_model_falls_back_to_anthropic():
    assert search_usd_per_call("gpt-7-omega") == 0.010


def test_grounded_gemini_call_prices_search_at_gemini_rate():
    # The web_anchor shape after the 2026-08 migration: a small prompt, a small
    # answer, and the retrieved context NOT billed as input.
    cost = compute_call_cost_usd(
        "gemini-3.7-flash",
        {"input_tokens": 500, "output_tokens": 200, "web_searches": 1},
    )
    expected = round(500 / 1_000_000 * 0.75 + 200 / 1_000_000 * 3.75 + 0.014, 6)
    assert cost == expected
