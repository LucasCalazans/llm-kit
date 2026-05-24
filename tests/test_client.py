"""LLMClient — unit-level coverage of pure helpers + integration with telemetry.

We don't exercise live LiteLLM here (that's an integration concern), but we do
verify that:
  - the constructor coalesces settings sources correctly
  - from_env() reads the standard env vars
  - the system-block translators behave for both providers
  - usage extraction handles cached vs uncached tokens correctly
  - _record_usage builds a CallRecord and fires the global callback registry
  - extract_anthropic_tool_citations walks LiteLLM's flattened shape
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llm_kit import (
    CallRecord,
    LLMClient,
    LLMResponse,
    LLMUsage,
    clear_usage_callbacks,
    extract_anthropic_tool_citations,
    set_usage_callback,
)


@pytest.fixture(autouse=True)
def _isolate_callbacks():
    clear_usage_callbacks()
    yield
    clear_usage_callbacks()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_constructor_with_explicit_kwargs():
    c = LLMClient(
        anthropic_api_key="sk-xyz",
        llm_model="claude-sonnet-4-5",
        llm_provider="anthropic",
    )
    assert c._settings.anthropic_api_key == "sk-xyz"
    assert c._settings.llm_model == "claude-sonnet-4-5"
    assert c._settings.llm_provider == "anthropic"


def test_constructor_reads_from_settings_obj():
    settings = SimpleNamespace(
        llm_provider="ollama",
        anthropic_api_key="",
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3",
        llm_model="",
    )
    c = LLMClient(settings)
    assert c._settings.llm_provider == "ollama"
    assert c._settings.ollama_base_url == "http://localhost:11434"


def test_constructor_kwargs_override_settings_obj():
    settings = SimpleNamespace(
        llm_provider="anthropic",
        anthropic_api_key="from-settings",
        ollama_base_url="",
        ollama_model="",
        llm_model="claude-sonnet-4-5",
    )
    c = LLMClient(settings, anthropic_api_key="from-kwarg")
    assert c._settings.anthropic_api_key == "from-kwarg"
    assert c._settings.llm_model == "claude-sonnet-4-5"  # settings_obj field preserved


def test_from_env_reads_standard_vars(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-haiku-4-5")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    c = LLMClient.from_env()
    assert c._settings.anthropic_api_key == "sk-env"
    assert c._settings.llm_provider == "anthropic"
    assert c._settings.llm_model == "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# System-block translation
# ---------------------------------------------------------------------------


def test_to_anthropic_system_string():
    out = LLMClient._to_anthropic_system("hello")
    assert out == [{"type": "text", "text": "hello"}]


def test_to_anthropic_system_blocks_with_cache():
    out = LLMClient._to_anthropic_system(
        [
            {"text": "first", "cache": "ephemeral"},
            {"text": "second"},
        ]
    )
    assert out == [
        {"type": "text", "text": "first", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "second"},
    ]


def test_to_anthropic_system_rejects_unknown_cache_mode():
    with pytest.raises(ValueError, match="unsupported cache mode"):
        LLMClient._to_anthropic_system([{"text": "x", "cache": "persistent"}])


def test_to_anthropic_system_none_returns_none():
    assert LLMClient._to_anthropic_system(None) is None


def test_to_ollama_system_flattens_blocks_preserving_order():
    text = LLMClient._to_ollama_system(
        [
            {"text": "alpha", "cache": "ephemeral"},  # cache flag silently ignored
            {"text": "beta"},
        ]
    )
    assert text == "alpha\n\nbeta"


def test_to_ollama_system_string_passthrough():
    assert LLMClient._to_ollama_system("just text") == "just text"


def test_to_ollama_system_none_returns_empty_string():
    assert LLMClient._to_ollama_system(None) == ""


# ---------------------------------------------------------------------------
# Text + usage extraction
# ---------------------------------------------------------------------------


def test_extract_text_string_content():
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))])
    assert LLMClient._extract_text(response) == "hello"


def test_extract_text_list_content_concatenates_text_blocks_and_skips_thinking():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=[
                        {"type": "thinking", "thinking": "internal"},
                        {"type": "text", "text": "visible "},
                        {"type": "text", "text": "answer"},
                    ]
                )
            )
        ]
    )
    assert LLMClient._extract_text(response) == "visible answer"


def test_extract_text_empty_when_no_choices():
    assert LLMClient._extract_text(SimpleNamespace(choices=[])) == ""


def test_extract_usage_subtracts_cache_when_text_tokens_absent():
    # LiteLLM's prompt_tokens is the SUM of uncached + cache_read + cache_write
    # when prompt_tokens_details.text_tokens is missing.
    usage_obj = SimpleNamespace(
        prompt_tokens=1500,
        completion_tokens=200,
        cache_read_input_tokens=500,
        cache_creation_input_tokens=300,
        prompt_tokens_details=None,
        server_tool_use=None,
    )
    response = SimpleNamespace(usage=usage_obj)
    usage = LLMClient._extract_usage(response)
    assert usage.input_tokens == 700  # 1500 - 500 - 300
    assert usage.output_tokens == 200
    assert usage.cache_read_tokens == 500
    assert usage.cache_write_tokens == 300


def test_extract_usage_prefers_text_tokens_when_present():
    details = SimpleNamespace(text_tokens=999)
    usage_obj = SimpleNamespace(
        prompt_tokens=1500,
        completion_tokens=200,
        cache_read_input_tokens=500,
        cache_creation_input_tokens=300,
        prompt_tokens_details=details,
        server_tool_use=None,
    )
    response = SimpleNamespace(usage=usage_obj)
    usage = LLMClient._extract_usage(response)
    assert usage.input_tokens == 999  # text_tokens wins over manual subtraction


def test_extract_usage_includes_web_search_count():
    usage_obj = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=50,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
        prompt_tokens_details=None,
        server_tool_use=SimpleNamespace(web_search_requests=3),
    )
    usage = LLMClient._extract_usage(SimpleNamespace(usage=usage_obj))
    assert usage.web_searches == 3


def test_extract_usage_no_usage_returns_zero():
    usage = LLMClient._extract_usage(SimpleNamespace(usage=None))
    assert usage == LLMUsage()


# ---------------------------------------------------------------------------
# Telemetry plumbing
# ---------------------------------------------------------------------------


def test_record_usage_fires_callback_with_pricing():
    captured: list[CallRecord] = []
    set_usage_callback(captured.append)

    c = LLMClient(anthropic_api_key="x", llm_model="claude-sonnet-4-5")
    c._record_usage(
        caller="my_feature",
        model="claude-sonnet-4-5",
        usage=LLMUsage(input_tokens=1_000_000, output_tokens=0),
    )

    assert len(captured) == 1
    rec = captured[0]
    assert rec.caller == "my_feature"
    assert rec.model == "claude-sonnet-4-5"
    assert rec.input_tokens == 1_000_000
    # 1M sonnet input tokens at $3/M = $3.00
    assert rec.cost_usd == 3.0


# ---------------------------------------------------------------------------
# Tool-citation extraction
# ---------------------------------------------------------------------------


def test_extract_citations_returns_empty_when_raw_is_none():
    resp = LLMResponse(text="x", model="m", usage=LLMUsage(), raw=None)
    assert extract_anthropic_tool_citations(resp) == []


def test_extract_citations_walks_web_search_results_and_dedupes():
    raw = MagicMock()
    message = MagicMock()
    message.provider_specific_fields = {
        "web_search_results": [
            {"content": [{"url": "https://a"}, {"url": "https://b"}]},
            {"content": [{"url": "https://a"}]},  # duplicate
        ],
        "citations": [
            [{"url": "https://c"}],
            {"url": "https://b"},  # duplicate, also tests non-list shape
        ],
    }
    raw.choices = [MagicMock(message=message)]
    resp = LLMResponse(text="x", model="m", usage=LLMUsage(), raw=raw)
    urls = extract_anthropic_tool_citations(resp)
    assert urls == ["https://a", "https://b", "https://c"]
