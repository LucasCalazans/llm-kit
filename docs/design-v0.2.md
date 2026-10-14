# llm-kit v0.2 — design proposal

Status: proposal, 2026-10-14 (checked against origin/main at 25e8be8). Written while integrating Gemini into a second consumer
(Estúdio do Corretor), next to Prism.

## Why v0.1 falls short

v0.1 serves Prism's original shape well (free-text completions, Anthropic first). A second consumer
that needs typed extraction on Gemini hits these limits:

1. **Structured output is not real.** `complete()` returns `text`. `json_schema` is enforced only on
   Ollama (`client.py`, docstring of `complete`); Anthropic ignores it, Gemini only gets
   `json_mode` (any JSON). Every consumer re-implements fence stripping, `json.loads` and
   validation — Prism has `_strip_json_fence` and at least four "parse the LLM JSON" helpers, while
   33 of its call sites pass a `json_schema` that silently does nothing outside Ollama.
2. **Pricing can still silently lie.** Since 25e8be8 the table has Gemini rows (3.1-flash-lite,
   3.5/3.6/3.7-flash, 3.1-pro) and counts image tokens, but any model without a row still falls back
   to the Sonnet `default` row ($3 / $15 per MTok): a `gemini-3.8-flash` call ($0.75 / $3.75) is
   recorded 4x too expensive, with no warning. Price changes are comments, not effective dates (the
   gemini-3.7/3.8-flash introductory rates double on 2027-01-01), and changing a price means a
   release plus a bump in every consumer.
3. **Provider knobs are missing.** No `temperature`, no Gemini 3 `thinking_level`, no image parts
   with `media_resolution` (the main cost lever for vision), no separate count of reasoning tokens
   (billed as output; through LiteLLM the mapping depends on the version).
4. **Telemetry carries Prism's domain.** `CallRecord` has `video_id` / `channel_slug`; the README
   suggests repurposing them. Other consumers need their own attribution (`workspace_id`,
   `listing_id`, `purpose`).
5. **Providers are tangled in one 724-line client** (`if provider == ...` per branch). Adding one
   means editing the core; Ollama already bypasses LiteLLM.
6. **Heavy, fragile install.** LiteLLM + OpenAI + Anthropic SDKs come with every install, even for a
   Gemini-only consumer. Prism pins `litellm==1.90.2` + `openai==2.44.0` after the 2026-08-17
   incompatibility.
7. **Async only.** Sync callers (job workers, sync web routes) need `asyncio.run` wrappers.

Worth keeping as is: the prompt registry (Markdown + YAML frontmatter, declared variables, hot
reload), callback-based telemetry, per-call cost, `retry-after`-aware retries, boot-time validation.

## Proposed structure

A small core, providers as adapters, typed output and pricing as first-class features.

```
llm_kit/
  types.py        Request / Response[T] shared by all providers:
                  - messages with text and image parts (image parts carry a resolution hint)
                  - options: max_tokens, temperature, reasoning ("low" | "medium" | "high" | token budget)
                  - Usage: input, output, reasoning, cache_read, cache_write, tool_calls
                  - Response: .text, .parsed (validated object), .usage, .cost_usd, .latency_ms, .raw
  client.py       complete() and complete_structured(schema=SomeModel), sync and async;
                  models addressed as "gemini:gemini-3.8-flash", "anthropic:claude-sonnet-4-5", "ollama:gemma4:12b"
  providers/      one module per provider behind the same Protocol; each declares capabilities
    base.py         Provider protocol + Capabilities (native_schema, vision, reasoning kind, caching)
    gemini.py       official google-genai SDK: native response_schema, thinking_level, media_resolution,
                    thoughts_token_count
    anthropic.py    official anthropic SDK: schema through a forced tool call; cache_control blocks
    ollama.py       httpx /api/chat with `format`
    litellm.py      optional catch-all
  structured.py   for providers without native schema enforcement: validate, and on a validation error
                  send the error back once for a repair; returns Response[T] or raises StructuredOutputError
  pricing/        data-driven price table (YAML) with effective dates, overridable by the consumer;
                  an unknown model yields cost_usd=None plus a warning (never another model's rates)
  telemetry.py    CallRecord with generic `attributes: dict[str, str]`; `with llm_context(**attrs)`;
                  failures are recorded too (error type, latency)
  prompts/        unchanged
  testing.py      FakeProvider with scripted responses, so consumers test without network
```

### Packaging

Extras per provider: `llm-kit[gemini]`, `llm-kit[anthropic]`, `llm-kit[ollama]`, `llm-kit[litellm]`.
Each consumer installs only what it uses; the LiteLLM/OpenAI pin problem stops leaking into
consumers that do not need them.

### Compatibility with Prism

- `LLMClient.complete(caller=..., messages=..., system=..., json_schema=..., provider=...,
  thinking_budget=...)` stays as a shim over the new core (deprecated in 0.2, removed in 0.3).
- `CallRecord.video_id` / `channel_slug` stay as properties reading `attributes`.
- Prism migrates its 33 `json_schema` call sites to `complete_structured` gradually, deleting its
  JSON-parsing helpers. Area routing (`llm_policy`) stays in Prism, where it belongs.

## Rollout

1. Core types, provider protocol, Gemini adapter, `complete_structured`, generic attributes,
   dated pricing, sync API, `testing.FakeProvider`. v0.1 shim keeps Prism green.
2. Anthropic adapter on the official SDK; LiteLLM becomes an optional extra.
3. Prism call sites move to `complete_structured`; shim removed in 0.3.

## Interim

Estúdio do Corretor integrates Gemini through a ~100-line internal module shaped like this API
(`complete_structured`, attributes, dated pricing), so switching to v0.2 later is mechanical.
