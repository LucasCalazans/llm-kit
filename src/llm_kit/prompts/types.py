"""Pydantic models for prompt metadata and rendered specs."""

from __future__ import annotations

import string
from typing import Literal

from pydantic import BaseModel, Field


class PromptMeta(BaseModel):
    """Frontmatter contract for a prompt file.

    Only ``name`` is required; everything else has sensible defaults so a
    quick experimental prompt can ship without ceremony. Validation of
    rendering variables happens in :class:`PromptSpec.render`, not here.
    """

    name: str
    version: int = 1
    description: str = ""
    model_hint: str = ""
    max_tokens_hint: int = 0
    cache: Literal["ephemeral"] | None = None
    variables: list[str] = Field(default_factory=list)


class PromptSpec(BaseModel):
    """A loaded prompt — frontmatter metadata plus the body template.

    The body is rendered with ``str.format()``; variables declared in the
    frontmatter are validated against the kwargs passed to :meth:`render`,
    so a typo in the call site fails loudly instead of silently dropping
    a value into the prompt.
    """

    meta: PromptMeta
    body: str
    source_path: str = ""

    @property
    def name(self) -> str:
        return self.meta.name

    @property
    def version(self) -> int:
        return self.meta.version

    @property
    def cache(self) -> str | None:
        return self.meta.cache

    @property
    def model_hint(self) -> str:
        return self.meta.model_hint

    def render(self, **kwargs: object) -> str:
        declared = set(self.meta.variables)
        provided = set(kwargs.keys())

        missing = declared - provided
        if missing:
            raise ValueError(f"prompt {self.meta.name!r}: missing required variables: {sorted(missing)}")

        extra = provided - declared
        if extra:
            raise ValueError(
                f"prompt {self.meta.name!r}: unexpected variables: {sorted(extra)} (declared: {sorted(declared)})"
            )

        # If the prompt declares no variables, treat it as plain text and
        # do not invoke ``str.format``. This prevents accidental KeyError
        # on JSON examples that contain literal ``{`` / ``}`` without
        # forcing every author to escape them.
        if not declared:
            return self.body

        try:
            return self.body.format(**kwargs)
        except KeyError as exc:
            raise ValueError(
                f"prompt {self.meta.name!r}: body references {exc!s} but it was not provided "
                f"(declared variables: {sorted(declared)}). "
                f"Did you forget to escape a literal '{{' as '{{{{' in the prompt body?"
            ) from exc
        except IndexError as exc:
            raise ValueError(
                f"prompt {self.meta.name!r}: positional placeholder in body — use named "
                f"variables only ({{name}}, not {{0}})."
            ) from exc

    def declared_field_names(self) -> set[str]:
        """Field names actually referenced by ``{...}`` in the body.

        Useful for tests that want to assert frontmatter ``variables`` and
        the body stay in sync.
        """
        names: set[str] = set()
        for _literal, field, _spec, _conv in string.Formatter().parse(self.body):
            if field is not None:
                root = field.split(".", 1)[0].split("[", 1)[0]
                if root:
                    names.add(root)
        return names
