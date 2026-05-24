"""Pricing math + family-prefix resolution."""

from __future__ import annotations

from llm_kit.pricing import compute_call_cost_usd, pricing_per_mtok, summarize_costs


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
