"""Telemetry — callback registry, contextvars, pipeline tracker."""

from __future__ import annotations

import pytest

from llm_kit import (
    CallRecord,
    clear_usage_callbacks,
    get_call_context,
    make_pipeline_usage_tracker,
    reset_call_context,
    reset_usage_tracker,
    set_call_context,
    set_usage_callback,
    set_usage_tracker,
    update_video_id,
)
from llm_kit.telemetry import _fire_usage_callback


@pytest.fixture(autouse=True)
def _isolate_callbacks():
    """Every test starts with an empty callback registry."""
    clear_usage_callbacks()
    yield
    clear_usage_callbacks()


def _make_record(**overrides) -> CallRecord:
    base = {
        "caller": "test",
        "model": "claude-sonnet-4-5",
        "input_tokens": 100,
        "output_tokens": 50,
        "cost_usd": 0.0015,
    }
    base.update(overrides)
    return CallRecord(**base)


def test_set_usage_callback_fires_on_record():
    captured: list[CallRecord] = []
    set_usage_callback(captured.append)
    _fire_usage_callback(_make_record())
    assert len(captured) == 1
    assert captured[0].caller == "test"


def test_multiple_callbacks_fire_in_order():
    order: list[str] = []
    set_usage_callback(lambda r: order.append("a"))
    set_usage_callback(lambda r: order.append("b"))
    _fire_usage_callback(_make_record())
    assert order == ["a", "b"]


def test_failing_callback_is_swallowed_and_does_not_block_others():
    captured: list[CallRecord] = []

    def boom(_: CallRecord) -> None:
        raise RuntimeError("simulated callback failure")

    set_usage_callback(boom)
    set_usage_callback(captured.append)
    _fire_usage_callback(_make_record())  # must not raise
    assert len(captured) == 1


def test_call_context_round_trip():
    tokens = set_call_context(
        video_id="vid-123",
        channel_slug="my-channel",
        feature_prefix="viral",
    )
    try:
        ctx = get_call_context()
        assert ctx == {
            "video_id": "vid-123",
            "channel_slug": "my-channel",
            "feature_prefix": "viral",
        }
    finally:
        reset_call_context(tokens)
    # After reset, all values are back to None.
    assert get_call_context() == {"video_id": None, "channel_slug": None, "feature_prefix": None}


def test_update_video_id_mutates_current_context():
    tokens = set_call_context(video_id="temp-uuid")
    try:
        update_video_id("final-slug")
        assert get_call_context()["video_id"] == "final-slug"
    finally:
        reset_call_context(tokens)


def test_record_carries_context_fields_when_passed_explicitly():
    captured: list[CallRecord] = []
    set_usage_callback(captured.append)
    _fire_usage_callback(_make_record(video_id="v-1", channel_slug="c-1", feature_prefix="p"))
    rec = captured[0]
    assert rec.video_id == "v-1"
    assert rec.channel_slug == "c-1"
    assert rec.feature_prefix == "p"


def test_pipeline_tracker_accumulates_per_model():
    tracker, summary = make_pipeline_usage_tracker()
    token = set_usage_tracker(tracker)
    try:
        _fire_usage_callback(
            _make_record(model="claude-sonnet-4-5", input_tokens=1000, output_tokens=500, cost_usd=0.0)
        )
        _fire_usage_callback(_make_record(model="claude-sonnet-4-5", input_tokens=200, output_tokens=100, cost_usd=0.0))
        _fire_usage_callback(_make_record(model="claude-haiku-4-5", input_tokens=2000, output_tokens=400, cost_usd=0.0))
    finally:
        reset_usage_tracker(token)

    s = summary()
    assert s["calls"] == 3
    assert s["input_tokens"] == 3200
    assert s["output_tokens"] == 1000
    assert "claude-sonnet-4-5" in s["by_model"]
    assert "claude-haiku-4-5" in s["by_model"]
    assert s["by_model"]["claude-sonnet-4-5"]["calls"] == 2


def test_pipeline_tracker_seed_preloads_buckets():
    seed = {
        "claude-sonnet-4-5": {
            "calls": 5,
            "input_tokens": 10_000,
            "output_tokens": 2_000,
            "total_usd": 0.123,
        }
    }
    tracker, summary = make_pipeline_usage_tracker(initial_by_model=seed)
    token = set_usage_tracker(tracker)
    try:
        _fire_usage_callback(
            _make_record(model="claude-sonnet-4-5", input_tokens=100, output_tokens=20, cost_usd=0.001)
        )
    finally:
        reset_usage_tracker(token)

    s = summary()
    assert s["by_model"]["claude-sonnet-4-5"]["calls"] == 6
    assert s["by_model"]["claude-sonnet-4-5"]["input_tokens"] == 10_100


def test_failing_tracker_is_swallowed():
    def boom(_: dict) -> None:
        raise RuntimeError("simulated tracker failure")

    token = set_usage_tracker(boom)
    try:
        _fire_usage_callback(_make_record())  # must not raise
    finally:
        reset_usage_tracker(token)


def test_callrecord_as_dict_includes_all_fields():
    rec = _make_record(video_id="v", channel_slug="c")
    d = rec.as_dict()
    assert d["caller"] == "test"
    assert d["video_id"] == "v"
    assert d["channel_slug"] == "c"
    assert d["model"] == "claude-sonnet-4-5"
