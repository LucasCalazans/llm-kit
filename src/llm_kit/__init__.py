"""llm-kit — provider-agnostic LLM client + prompt registry + pricing.

A small infrastructure layer for Python projects that talk to LLMs:
ships an Anthropic/Ollama client (via LiteLLM), a Markdown+YAML prompt
registry, USD pricing tables for cost estimation, and a pluggable
telemetry contract.

See README.md for the quick-start guide.
"""

from llm_kit.client import (
    LLMClient,
    LLMResponse,
    LLMUsage,
    extract_search_citations,
)
from llm_kit.config import SUPPORTED_PROVIDERS, resolve_model, validate_llm_config
from llm_kit.exceptions import (
    LLMConfigError,
    LLMError,
    LLMOverloadedError,
    LLMRateLimitError,
)
from llm_kit.pricing import (
    compute_call_cost_usd,
    pricing_per_mtok,
    search_usd_per_call,
    summarize_costs,
)
from llm_kit.prompts import PromptMeta, PromptRegistry, PromptSpec
from llm_kit.telemetry import (
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

__version__ = "0.1.0"

__all__ = [
    # Client
    "LLMClient",
    "LLMResponse",
    "LLMUsage",
    "extract_search_citations",
    # Config
    "SUPPORTED_PROVIDERS",
    "resolve_model",
    "validate_llm_config",
    # Exceptions
    "LLMError",
    "LLMConfigError",
    "LLMRateLimitError",
    "LLMOverloadedError",
    # Pricing
    "compute_call_cost_usd",
    "pricing_per_mtok",
    "search_usd_per_call",
    "summarize_costs",
    # Prompts
    "PromptRegistry",
    "PromptMeta",
    "PromptSpec",
    # Telemetry
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
