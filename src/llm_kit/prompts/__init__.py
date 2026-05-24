"""Prompt registry — Markdown files with YAML frontmatter loaded by name.

Usage in a consumer project:

    from pathlib import Path
    from llm_kit import PromptRegistry

    registry = PromptRegistry(root=Path(__file__).parent / "prompts")
    spec = registry.get("greetings.system")
    text = spec.render(name="world")
"""

from llm_kit.prompts.registry import PromptRegistry
from llm_kit.prompts.types import PromptMeta, PromptSpec

__all__ = ["PromptRegistry", "PromptMeta", "PromptSpec"]
