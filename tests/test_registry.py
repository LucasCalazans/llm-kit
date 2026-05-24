"""PromptRegistry — loading, rendering, caching, hot-reload, error paths."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from llm_kit import PromptRegistry, PromptSpec

FIXTURES = Path(__file__).parent / "fixtures" / "prompts" / "sample"


def _registry() -> PromptRegistry:
    return PromptRegistry(root=FIXTURES.parent)


def test_get_loads_spec_and_parses_frontmatter():
    reg = _registry()
    spec = reg.get("sample.hello")
    assert isinstance(spec, PromptSpec)
    assert spec.meta.name == "sample.hello"
    assert spec.meta.version == 1
    assert spec.meta.cache == "ephemeral"
    assert spec.meta.variables == ["name"]
    assert "Hello, {name}!" in spec.body


def test_render_substitutes_variables():
    text = _registry().render("sample.hello", name="world")
    assert text.strip() == "Hello, world!"


def test_render_missing_variable_raises_value_error():
    with pytest.raises(ValueError, match="missing required variables"):
        _registry().render("sample.hello")


def test_render_unexpected_variable_raises_value_error():
    with pytest.raises(ValueError, match="unexpected variables"):
        _registry().render("sample.hello", name="x", foo="bar")


def test_no_variable_prompt_returns_body_verbatim_with_literal_braces():
    text = _registry().render("sample.plain")
    assert "{curly}" in text


def test_dotted_name_required():
    with pytest.raises(ValueError, match="must be 'area.name'"):
        _registry().get("not_dotted")


def test_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        _registry().get("sample.does_not_exist")


def test_cache_returns_same_instance_within_process():
    reg = _registry()
    a = reg.get("sample.hello")
    b = reg.get("sample.hello")
    assert a is b


def test_clear_drops_cache():
    reg = _registry()
    a = reg.get("sample.hello")
    reg.clear()
    b = reg.get("sample.hello")
    assert a is not b


def test_auto_reload_picks_up_mtime_change(tmp_path, monkeypatch):
    # Copy the fixture into a writable temp dir so we can mutate it.
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    target = sample_dir / "hot.md"
    target.write_text(
        "---\nname: sample.hot\nvariables: [x]\n---\nv1 {x}\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("PROMPTS_AUTO_RELOAD", "1")
    reg = PromptRegistry(root=tmp_path)

    first = reg.render("sample.hot", x="a")
    assert first.strip() == "v1 a"

    # Wait long enough to guarantee mtime granularity on every filesystem
    # (NTFS-over-9p in WSL is coarse), then rewrite.
    time.sleep(1.1)
    target.write_text(
        "---\nname: sample.hot\nvariables: [x]\n---\nv2 {x}\n",
        encoding="utf-8",
    )
    # Force-bump mtime in case the FS resolution is too coarse.
    new_mtime = time.time()
    os.utime(target, (new_mtime, new_mtime))

    second = reg.render("sample.hot", x="a")
    assert second.strip() == "v2 a"


def test_invalid_frontmatter_raises_value_error(tmp_path):
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "broken.md").write_text("no frontmatter here\n", encoding="utf-8")
    reg = PromptRegistry(root=tmp_path)
    with pytest.raises(ValueError, match="missing the YAML frontmatter block"):
        reg.get("bad.broken")


def test_declared_field_names_matches_body_placeholders():
    spec = _registry().get("sample.hello")
    assert spec.declared_field_names() == {"name"}
