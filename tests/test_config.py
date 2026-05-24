"""Config — provider whitelist + boot-time validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_kit import (
    SUPPORTED_PROVIDERS,
    LLMConfigError,
    resolve_model,
    validate_llm_config,
)


def _settings(**overrides) -> SimpleNamespace:
    base = {
        "llm_provider": "anthropic",
        "anthropic_api_key": "sk-test",
        "ollama_base_url": "",
        "ollama_model": "",
        "llm_model": "claude-sonnet-4-5",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_supported_providers_includes_anthropic_and_ollama():
    assert "anthropic" in SUPPORTED_PROVIDERS
    assert "ollama" in SUPPORTED_PROVIDERS


def test_resolve_model_anthropic_returns_llm_model():
    s = _settings()
    assert resolve_model(s) == "claude-sonnet-4-5"


def test_resolve_model_ollama_returns_ollama_model():
    s = _settings(llm_provider="ollama", ollama_model="llama3")
    assert resolve_model(s) == "llama3"


def test_resolve_model_per_call_provider_override_wins():
    s = _settings(ollama_model="llama3")
    # Globally anthropic, but a per-call override should resolve as ollama.
    assert resolve_model(s, provider="ollama") == "llama3"


def test_validate_passes_for_well_configured_anthropic():
    validate_llm_config(_settings())  # no raise


def test_validate_rejects_unknown_provider():
    s = _settings(llm_provider="anthopic")  # typo
    with pytest.raises(LLMConfigError, match="not enabled"):
        validate_llm_config(s)


def test_validate_rejects_anthropic_without_api_key():
    s = _settings(anthropic_api_key="")
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        validate_llm_config(s)


def test_validate_rejects_ollama_without_base_url():
    s = _settings(llm_provider="ollama", ollama_base_url="", ollama_model="llama3")
    with pytest.raises(LLMConfigError, match="OLLAMA_BASE_URL"):
        validate_llm_config(s)


def test_validate_rejects_ollama_without_model():
    s = _settings(llm_provider="ollama", ollama_base_url="http://localhost:11434", ollama_model="")
    with pytest.raises(LLMConfigError, match="OLLAMA_MODEL"):
        validate_llm_config(s)


def test_validate_rejects_empty_model_string():
    s = _settings(llm_model="")
    with pytest.raises(LLMConfigError, match="LLM_MODEL is empty"):
        validate_llm_config(s)
