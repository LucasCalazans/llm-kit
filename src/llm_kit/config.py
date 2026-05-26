"""Provider whitelist + boot-time validation for the LLM abstraction.

Adding a new provider is two changes: append the slug to
:data:`SUPPORTED_PROVIDERS`, then handle it in :class:`llm_kit.client.LLMClient`.
The whitelist exists so a typo (``LLM_PROVIDER=anthopic``) fails fast at boot
instead of mid-pipeline.

``settings_obj`` is duck-typed: any object exposing the expected attributes
(``llm_provider``, ``anthropic_api_key``, ``ollama_base_url``, ``ollama_model``,
``llm_model``) works. Pydantic settings, plain dataclasses, ``argparse``
namespaces, and ``types.SimpleNamespace`` are all valid. The consumer owns
where settings come from.
"""

from __future__ import annotations

from llm_kit.exceptions import LLMConfigError

SUPPORTED_PROVIDERS = ("anthropic", "ollama", "gemini")


def resolve_model(settings_obj, provider: str | None = None) -> str:
    """Return the active model string for the given (or globally-configured)
    provider. Per-call overrides flow through the ``model=`` kwarg on
    :meth:`LLMClient.complete`, not through here."""
    active = provider or getattr(settings_obj, "llm_provider", "anthropic")
    if active == "ollama":
        return getattr(settings_obj, "ollama_model", "") or ""
    # Anthropic and Gemini both read from llm_model; the provider prefix
    # (anthropic/… or gemini/…) is added inside the per-provider call path.
    return getattr(settings_obj, "llm_model", "") or ""


def validate_llm_config(settings_obj) -> None:
    """Boot-time check: provider is whitelisted and credentials are present.

    Raises :class:`LLMConfigError` with an actionable message. Wire this into
    your app's startup so misconfiguration shows up immediately, not on the
    first user request.

    For Ollama, the global default may stay Anthropic — Ollama is enabled
    per-call, not necessarily as a global swap. We therefore only check
    Ollama settings when ``LLM_PROVIDER=ollama`` is the global default.
    """
    provider = getattr(settings_obj, "llm_provider", "anthropic")
    if provider not in SUPPORTED_PROVIDERS:
        raise LLMConfigError(
            f"LLM_PROVIDER={provider!r} is not enabled. "
            f"Supported: {list(SUPPORTED_PROVIDERS)}. "
            f"To add another provider, extend SUPPORTED_PROVIDERS in llm_kit/config.py."
        )
    if provider == "anthropic" and not getattr(settings_obj, "anthropic_api_key", ""):
        raise LLMConfigError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY to be set.")
    if provider == "gemini" and not getattr(settings_obj, "gemini_api_key", ""):
        raise LLMConfigError("LLM_PROVIDER=gemini requires GEMINI_API_KEY to be set.")
    if provider == "ollama":
        if not getattr(settings_obj, "ollama_base_url", ""):
            raise LLMConfigError("LLM_PROVIDER=ollama requires OLLAMA_BASE_URL to be set.")
        if not getattr(settings_obj, "ollama_model", ""):
            raise LLMConfigError("LLM_PROVIDER=ollama requires OLLAMA_MODEL to be set.")
    if not resolve_model(settings_obj):
        raise LLMConfigError("LLM_MODEL is empty: set it in your environment (e.g. gemini-2.5-flash).")
