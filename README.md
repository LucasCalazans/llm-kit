# llm-kit

Provider-agnostic LLM client + prompt registry + pricing — a small infrastructure layer for Python projects that talk to LLMs.

The package ships **infra** (client, registry, telemetry, pricing); each consumer project brings its own **content** (`.md` prompts, persistence callback, app settings).

---

## Install (Git dependency)

`pip` accepts Git dependencies directly in `pyproject.toml` via PEP 508 — the Python equivalent of `"llm-kit": "github:owner/llm-kit#v0.1.0"` in npm.

```toml
# consumer project's pyproject.toml
dependencies = [
    "llm-kit @ git+https://github.com/LucasCalazans/llm-kit.git@v0.1.0",
    # other deps...
]
```

Then:

```bash
pip install -e .   # or pip install . — any install resolves the git dep
```

Works with public or private repos (use `git+ssh://git@github.com/...` for SSH auth on private). To update: bump the tag in `pyproject.toml` and re-run `pip install -e .`.

### Dev mode — local editable install

While iterating on `llm-kit` itself, install editable against the local checkout:

```bash
pip install -e ~/projects/libs/llm-kit
```

Changes to the package show up immediately in the consumer project — no tag, no reinstall. Once stable, cut a tag (`git tag v0.1.1 && git push --tags`) and switch back to the tagged dep in `pyproject.toml`.

### Versioning workflow

1. Edit code in your `llm-kit` checkout.
2. `git commit` + `git tag v0.1.X && git push --tags`.
3. In each consumer project: bump the tag in `pyproject.toml` and run `pip install -e .`.

Same UX as the npm flow — no PyPI release, no `twine upload`.

---

## Quick start

```python
from pathlib import Path
from llm_kit import LLMClient, PromptRegistry, set_usage_callback, CallRecord

# 1. Point a registry at your project's prompt directory.
registry = PromptRegistry(root=Path(__file__).parent / "prompts")

# 2. (Optional) Register a callback to capture per-call usage / cost.
def on_call(record: CallRecord) -> None:
    print(f"{record.model}: ${record.cost_usd:.4f}")

set_usage_callback(on_call)

# 3. Use the client.
client = LLMClient.from_env()   # reads ANTHROPIC_API_KEY, LLM_PROVIDER, etc.
resp = await client.complete(
    caller="my_feature",
    model="claude-sonnet-4-5",
    messages=[{"role": "user", "content": "Hi"}],
    system=registry.render("chat.system"),
    max_tokens=512,
)
print(resp.text, resp.usage)
```

---

## Integration pattern (recommended)

This is the layout we use in real production code. Copy-paste into any new project; it's three small files.

### 1. Bootstrap module — single client + telemetry wiring

```python
# myapp/services/llm_bootstrap.py
import logging
from llm_kit import CallRecord, LLMClient, set_usage_callback
from myapp.settings import settings

logger = logging.getLogger(__name__)

# Module-level singleton. The constructor is cheap (just stashes settings).
llm_client = LLMClient(settings)

_callback_registered = False


def _persist_call(record: CallRecord) -> None:
    """Map a CallRecord into your domain row (here: an `llm_usage_log` table)."""
    try:
        from myapp.db import usage_repo

        feature = record.caller
        if record.feature_prefix:
            feature = f"{record.feature_prefix}.{feature}"

        usage_repo.log(
            feature=feature,
            model=record.model,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            cache_read_tokens=record.cache_read_tokens,
            cache_write_tokens=record.cache_write_tokens,
            web_searches=record.web_searches,
            total_usd=record.cost_usd,
            request_id=record.video_id,        # repurpose attribution fields
            tenant_slug=record.channel_slug,   # to whatever your app needs
        )
    except Exception:
        logger.exception("llm_usage_log write failed (non-fatal)")


def register_telemetry_callback() -> None:
    """Idempotent — call once per entry point on startup."""
    global _callback_registered
    if _callback_registered:
        return
    set_usage_callback(_persist_call)
    _callback_registered = True
```

### 2. Wire on startup — web lifespan + CLI entry

```python
# myapp/web/app.py — inside the FastAPI lifespan
from myapp.services.llm_bootstrap import register_telemetry_callback
register_telemetry_callback()
```

```python
# myapp/cli.py — at the top of your CLI entry group
from myapp.services.llm_bootstrap import register_telemetry_callback
register_telemetry_callback()
```

### 3. Prompt registry — instance with local root

```python
# myapp/prompts/__init__.py
from pathlib import Path
from llm_kit import PromptRegistry, PromptMeta, PromptSpec

# App-specific routing flags live here, not in the package. Example: some
# prompt areas could route to a local Ollama instead of Anthropic.
AREA_LLM_FLAGS: dict[str, dict[str, bool]] = {
    "classifier": {"use_local": False},
    "metadata":   {"use_local": False},
}

def area_uses_local(area: str) -> bool:
    return AREA_LLM_FLAGS.get(area, {}).get("use_local", False)

# The registry points at the .md files that live alongside this module.
registry = PromptRegistry(root=Path(__file__).resolve().parent)
```

The `.md` files live in `myapp/prompts/<area>/<name>.md` — the package only ships the loader, parser, and hot-reload logic.

### 4. Usage in any call site

```python
from myapp.prompts import registry
from myapp.services.llm_bootstrap import llm_client

resp = await llm_client.complete(
    caller="chat.reply",
    model="claude-sonnet-4-5",
    messages=[{"role": "user", "content": user_input}],
    system=[
        {"text": registry.render("chat.system"), "cache": "ephemeral"},
    ],
    max_tokens=2048,
)
print(resp.text, resp.usage)
```

---

## Public API

```python
from llm_kit import (
    # Client
    LLMClient,           # LLMClient(settings_obj) or LLMClient.from_env()
    LLMResponse,         # .text, .model, .usage, .raw
    LLMUsage,            # input/output/cache_read/cache_write/web_searches tokens
    extract_anthropic_tool_citations,  # URLs from the web_search tool

    # Exceptions (catch these, not vendor-specific ones)
    LLMError, LLMConfigError, LLMRateLimitError, LLMOverloadedError,

    # Config
    SUPPORTED_PROVIDERS, resolve_model, validate_llm_config,

    # Pricing (USD per 1M tokens — Claude Sonnet/Opus/Haiku 4.x)
    compute_call_cost_usd, pricing_per_mtok, summarize_costs,

    # Prompts (Markdown + YAML frontmatter; hot-reload via PROMPTS_AUTO_RELOAD=1)
    PromptRegistry, PromptMeta, PromptSpec,

    # Telemetry
    CallRecord,
    set_usage_callback, clear_usage_callbacks,
    set_call_context, reset_call_context, get_call_context, update_video_id,
    set_usage_tracker, reset_usage_tracker, make_pipeline_usage_tracker,
)
```

### Telemetry — full flow

```
LLMClient.complete()
    ↓ (call succeeds)
build CallRecord (with cost_usd via pricing.py)
    ↓
for each registered callback: callback(record)   ← your persistence here
    ↓
if a pipeline tracker is active: tracker(record.as_dict())   ← local accumulator
```

- `set_usage_callback(fn)` — global, stays until `clear_usage_callbacks()`. Use for DB writes, metrics, log lines.
- `set_call_context(video_id=..., channel_slug=..., feature_prefix=...)` — contextvar-based attribution; values land on the `CallRecord` automatically. Reset at the end of the operation. The field names are intentionally generic — repurpose them for whatever attribution your app needs.
- `make_pipeline_usage_tracker()` — returns `(tracker, get_summary)`. Activate with `set_usage_tracker(tracker)` at the start of a pipeline; call `get_summary()` at the end to get totals + per-model breakdown.

Callbacks **must not raise** — exceptions are logged and swallowed so telemetry never blocks the pipeline.

---

## Client configuration

Three ways:

```python
# A) settings_obj (any object exposing the expected attributes)
from myapp.settings import settings   # pydantic-settings, SimpleNamespace, ...
client = LLMClient(settings)

# B) explicit kwargs (override settings_obj fields when passed together)
client = LLMClient(
    anthropic_api_key="sk-...",
    llm_model="claude-sonnet-4-5",
    llm_provider="anthropic",
    ollama_base_url="http://localhost:11434",
    ollama_model="qwen2.5:7b-instruct",
)

# C) from_env() — reads ANTHROPIC_API_KEY, LLM_PROVIDER, LLM_MODEL,
#    OLLAMA_BASE_URL, OLLAMA_MODEL
client = LLMClient.from_env()
```

For fail-fast boot:

```python
from llm_kit import validate_llm_config
validate_llm_config(settings)  # raises LLMConfigError if something crucial is missing
```

---

## Supported providers

| Provider | How to enable | Notes |
|---|---|---|
| `anthropic` (default) | `LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY` | Through LiteLLM. Supports prompt caching (`cache="ephemeral"`), extended thinking, web_search tool. |
| `ollama` | `LLM_PROVIDER=ollama` + `OLLAMA_BASE_URL` + `OLLAMA_MODEL` | Local. Talks directly to `/api/chat`; supports server-side JSON schema via `json_schema=` on `.complete()`. |

Adding OpenAI direct, Gemini, etc. means extending `SUPPORTED_PROVIDERS` in `llm_kit/config.py` and the routing inside `LLMClient.complete()`. LiteLLM already supports many providers — the work is whitelisting + translating their quirks.

---

## Developing the package

```bash
cd ~/projects/libs/llm-kit
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                          # 62 tests, ~5s
ruff check src tests            # lint
ruff format src tests           # format
```

Release workflow:

```bash
# After changes and green tests:
git add -A
git commit -m "feat: <description>"
git tag v0.1.X
git push && git push --tags
```

Then in each consumer project, bump the tag in `pyproject.toml` + `pip install -e .`.

---

## License

MIT — see [LICENSE](LICENSE).
