"""Telemetry contract for LLM calls — contextvars + callback registry.

This module is the single source of truth for the contextvars that carry
call attribution down through async layers (``video_id``, ``channel_slug``,
``feature_prefix``) and for the per-call ``CallRecord`` that gets dispatched
to every registered callback after a successful completion.

Consumer projects register callbacks at startup:

    from llm_kit import set_usage_callback, CallRecord

    def persist(record: CallRecord) -> None:
        my_db.insert("llm_usage", record.as_dict())

    set_usage_callback(persist)

Multiple callbacks may be registered; each is invoked in registration order.
Callback failures are logged and swallowed — telemetry never blocks the
pipeline.

Beyond the global callback registry, a per-pipeline tracker hook
(:func:`set_usage_tracker` / :func:`make_pipeline_usage_tracker`) lets a
single in-flight pipeline keep its own running totals without coupling to
the global registry.
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CallRecord:
    """Normalized record of one successful LLM call.

    Fired through the callback registry after pricing. Consumer projects that
    want to persist usage convert this into their domain row inside the
    callback (e.g. an ``llm_usage_log`` table, a Prometheus counter, a log
    line — the package stays agnostic).
    """

    caller: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    web_searches: int = 0
    cost_usd: float = 0.0
    video_id: str | None = None
    channel_slug: str | None = None
    feature_prefix: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Contextvars — call attribution carried through async layers
# ---------------------------------------------------------------------------

_video_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("llm_kit._video_id_ctx", default=None)
_channel_slug_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_kit._channel_slug_ctx", default=None
)
_feature_prefix_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llm_kit._feature_prefix_ctx", default=None
)

# Optional per-pipeline tracker hook. Distinct from the global callback
# registry: the tracker is meant for a single in-flight pipeline that wants
# its own running totals, then discards the contextvar token on exit.
_usage_tracker: contextvars.ContextVar[Callable[[dict], None] | None] = contextvars.ContextVar(
    "llm_kit._usage_tracker", default=None
)


def set_call_context(
    *,
    video_id: str | None = None,
    channel_slug: str | None = None,
    feature_prefix: str | None = None,
) -> dict[str, contextvars.Token]:
    """Set call attribution for the current async context.

    Returns the token bundle to feed into :func:`reset_call_context`.
    Pipelines call this once at the entry and reset on the way out; nested
    helpers don't need to know about the values to keep telemetry working.
    """
    return {
        "video_id": _video_id_ctx.set(video_id),
        "channel_slug": _channel_slug_ctx.set(channel_slug),
        "feature_prefix": _feature_prefix_ctx.set(feature_prefix),
    }


def reset_call_context(tokens: dict[str, contextvars.Token]) -> None:
    if "video_id" in tokens:
        _video_id_ctx.reset(tokens["video_id"])
    if "channel_slug" in tokens:
        _channel_slug_ctx.reset(tokens["channel_slug"])
    if "feature_prefix" in tokens:
        _feature_prefix_ctx.reset(tokens["feature_prefix"])


def update_video_id(new_video_id: str) -> None:
    """Roll-set the video_id mid-pipeline (e.g. after slug rename).

    No token returned — the entry's ``set_call_context`` token will reset
    everything at pipeline exit. Calls happening AFTER this update get the
    new id; calls before keep the temporary uuid.
    """
    _video_id_ctx.set(new_video_id)


def get_call_context() -> dict[str, str | None]:
    """Read-only snapshot of the current attribution context. Mainly for tests
    and for callback implementations that want to inspect the values."""
    return {
        "video_id": _video_id_ctx.get(),
        "channel_slug": _channel_slug_ctx.get(),
        "feature_prefix": _feature_prefix_ctx.get(),
    }


def set_usage_tracker(tracker: Callable[[dict], None] | None) -> contextvars.Token:
    return _usage_tracker.set(tracker)


def reset_usage_tracker(token: contextvars.Token) -> None:
    _usage_tracker.reset(token)


# ---------------------------------------------------------------------------
# Global callback registry — fires after every successful LLM call
# ---------------------------------------------------------------------------

_callbacks: list[Callable[[CallRecord], None]] = []


def set_usage_callback(callback: Callable[[CallRecord], None]) -> None:
    """Append a callback to the global registry.

    Every successful LLM call dispatches the resulting :class:`CallRecord`
    to every callback. Callbacks must be cheap and non-raising — exceptions
    are logged at WARNING level and swallowed.

    Use this to wire persistence (write to a usage table), metrics (push to
    Prometheus / DataDog), or just a debug logger. Multiple callbacks are
    invoked in registration order.
    """
    _callbacks.append(callback)


def clear_usage_callbacks() -> None:
    """Drop every registered callback. Mainly useful in tests."""
    _callbacks.clear()


def _fire_usage_callback(record: CallRecord) -> None:
    """Dispatch ``record`` to every callback and the per-pipeline tracker.

    Called from :class:`llm_kit.client.LLMClient`. Pipelines that want a
    running per-model summary set the tracker via :func:`set_usage_tracker`;
    callbacks set globally via :func:`set_usage_callback` fire for every
    call regardless of pipeline context.

    Failures are logged at WARNING level and swallowed so telemetry never
    blocks generation.
    """
    payload = record.as_dict()

    for cb in list(_callbacks):
        try:
            cb(record)
        except Exception:  # noqa: BLE001
            logger.exception("llm_kit usage callback raised (non-fatal)")

    tracker = _usage_tracker.get()
    if tracker is not None:
        try:
            tracker(payload)
        except Exception:  # noqa: BLE001
            logger.exception("llm_kit usage tracker raised (non-fatal)")


def make_pipeline_usage_tracker(
    initial_by_model: dict[str, dict[str, Any]] | None = None,
) -> tuple[Callable[[dict], None], Callable[[], dict[str, Any]]]:
    """Return ``(tracker, get_summary)`` — boilerplate every pipeline used to inline.

    Use as::

        tracker, get_summary = make_pipeline_usage_tracker()
        tracker_token = set_usage_tracker(tracker)
        try:
            ...  # pipeline body
            project.cost_summary = get_summary()
        finally:
            reset_usage_tracker(tracker_token)

    The tracker accumulates each call's ``cost_usd`` (already computed before
    the callback fires) into a per-model bucket. ``get_summary()`` returns
    the legacy-shape dict (with a ``by_model`` breakdown).

    Pass ``initial_by_model`` to seed the accumulator with prior per-model
    totals (e.g. a shared fetch cost reused across two language variants,
    or a pre-script enrichment step whose cost should be attributed to a
    later pipeline). The seed must already follow the per-model bucket
    shape that this tracker emits — typically the ``by_model`` field from
    a previous ``get_summary()`` call.
    """
    by_model: dict[str, dict[str, Any]] = {}
    if initial_by_model:
        for model, bucket in initial_by_model.items():
            by_model[model] = {
                "calls": int(bucket.get("calls", 0) or 0),
                "input_tokens": int(bucket.get("input_tokens", 0) or 0),
                "output_tokens": int(bucket.get("output_tokens", 0) or 0),
                "cache_read_tokens": int(bucket.get("cache_read_tokens", 0) or 0),
                "cache_write_tokens": int(bucket.get("cache_write_tokens", 0) or 0),
                "web_searches": int(bucket.get("web_searches", 0) or 0),
                "total_usd": float(bucket.get("total_usd", 0.0) or 0.0),
            }

    def _tracker(usage: dict) -> None:
        model = (usage.get("model") or "") or "(unknown)"
        bucket = by_model.setdefault(
            model,
            {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "web_searches": 0,
                "total_usd": 0.0,
            },
        )
        bucket["calls"] += 1
        for key in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "web_searches"):
            bucket[key] += int(usage.get(key, 0) or 0)
        bucket["total_usd"] += float(usage.get("cost_usd", 0.0) or 0.0)

    def _summary() -> dict[str, Any]:
        from llm_kit.pricing import summarize_costs

        return summarize_costs(by_model)

    return _tracker, _summary


__all__ = [
    "CallRecord",
    "set_usage_callback",
    "clear_usage_callbacks",
    "set_call_context",
    "reset_call_context",
    "get_call_context",
    "update_video_id",
    "set_usage_tracker",
    "reset_usage_tracker",
    "make_pipeline_usage_tracker",
]
