"""Lazy, in-memory registry for Markdown prompts with YAML frontmatter.

Files live under ``<root>/<area>/<name>.md`` and are addressed by dotted
name (``chat.system`` → ``<root>/chat/system.md``). The
``root`` directory is passed at construction time — there is no module-level
global, because each consumer project keeps its prompts in its own tree.

Hot-reload: when ``PROMPTS_AUTO_RELOAD=1`` (or ``true``) is in the
environment, the registry checks the file's mtime on each ``get()`` and
re-parses on change. Off by default — production caches the first parse
for the lifetime of the process.

Per-area routing flags (Anthropic vs. local Ollama, A/B testing slots,
feature gates) are intentionally NOT part of this registry. Consumer
projects own that policy; the registry only knows how to load and render.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path

import yaml

from llm_kit.prompts.types import PromptMeta, PromptSpec

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


def _parse_file(path: Path) -> PromptSpec:
    raw = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        raise ValueError(
            f"prompt file {path} is missing the YAML frontmatter block "
            f"(expected '---\\n<yaml>\\n---\\n<body>' at the top of the file)"
        )
    fm_raw, body = match.group(1), match.group(2)
    try:
        fm_data = yaml.safe_load(fm_raw) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"prompt file {path}: invalid YAML frontmatter: {exc}") from exc
    if not isinstance(fm_data, dict):
        raise ValueError(f"prompt file {path}: frontmatter must be a YAML mapping, got {type(fm_data).__name__}")

    meta = PromptMeta(**fm_data)
    return PromptSpec(meta=meta, body=body, source_path=str(path))


def _auto_reload_enabled() -> bool:
    return os.environ.get("PROMPTS_AUTO_RELOAD", "").lower() in ("1", "true", "yes")


class PromptRegistry:
    """Thread-safe lazy loader keyed by dotted name (``area.name``).

    Resolves to ``<root>/<area>/<name>.md``. The ``root`` is required and
    must point at the directory tree that holds the consumer project's
    prompt files.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._cache: dict[str, tuple[float, PromptSpec]] = {}
        self._lock = threading.RLock()

    def _path_for(self, name: str) -> Path:
        if "." not in name:
            raise ValueError(
                f"prompt name must be 'area.name' (got {name!r}); e.g. 'chat.system' resolves to chat/system.md"
            )
        area, _, leaf = name.partition(".")
        return self._root / area / f"{leaf}.md"

    def get(self, name: str, /) -> PromptSpec:
        path = self._path_for(name)
        if not path.exists():
            raise FileNotFoundError(f"prompt {name!r} not found at {path}")

        if _auto_reload_enabled():
            mtime = path.stat().st_mtime
            with self._lock:
                cached = self._cache.get(name)
                if cached is None or cached[0] != mtime:
                    spec = _parse_file(path)
                    self._cache[name] = (mtime, spec)
                    return spec
                return cached[1]

        with self._lock:
            cached = self._cache.get(name)
            if cached is not None:
                return cached[1]
            spec = _parse_file(path)
            # Store mtime even in the cached path so a later flip of
            # PROMPTS_AUTO_RELOAD picks up the right baseline.
            self._cache[name] = (path.stat().st_mtime, spec)
            return spec

    def render(self, name: str, /, **kwargs: object) -> str:
        return self.get(name).render(**kwargs)

    def clear(self) -> None:
        """Drop all cached specs. Mainly useful in tests."""
        with self._lock:
            self._cache.clear()
