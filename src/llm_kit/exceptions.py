"""Internal exception hierarchy for the LLM abstraction.

Callers should catch these instead of vendor-specific exceptions, so we can
swap providers in the future without touching call sites.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base class for all LLM-layer errors."""


class LLMConfigError(LLMError):
    """The LLM_PROVIDER / LLM_MODEL configuration is invalid or unsupported."""


class LLMRateLimitError(LLMError):
    """Provider returned 429. Already retried internally — surface to the caller
    only if every attempt failed."""


class LLMOverloadedError(LLMError):
    """Provider returned 529 (Anthropic-specific 'overloaded'). Same retry policy
    as :class:`LLMRateLimitError`."""
