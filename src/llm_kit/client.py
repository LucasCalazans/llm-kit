"""Unified LLM client — single ``complete()`` entry point used by call sites.

Internally routes through ``litellm.acompletion`` with the Anthropic provider
configured. The interface is provider-agnostic so adding Ollama or OpenAI in
the future is a whitelist + per-call ``provider=`` change, not a refactor.

System blocks accept a friendlier shape than Anthropic's raw API:

    system=[
        {"text": guidelines, "cache": "ephemeral"},
        {"text": system_prompt, "cache": "ephemeral"},
    ]

…which gets translated into the provider-specific ``cache_control`` payload
inside :meth:`LLMClient._anthropic_complete`.

Telemetry: every successful call fires
:func:`llm_kit.telemetry._fire_usage_callback`, which dispatches the
:class:`CallRecord` to every registered callback and to the per-pipeline
tracker (when set). Failures are swallowed — telemetry never blocks the
pipeline.

Configuration: ``LLMClient`` accepts settings either as a duck-typed
``settings_obj`` (any object exposing ``llm_provider``,
``anthropic_api_key``, ``ollama_base_url``, ``ollama_model``, ``llm_model``)
or via keyword arguments. The :meth:`from_env` classmethod reads the
standard env vars (``ANTHROPIC_API_KEY``, ``LLM_PROVIDER``,
``OLLAMA_BASE_URL``, ``OLLAMA_MODEL``, ``LLM_MODEL``) and is the recommended
entry point for most consumers.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import httpx
import litellm
from litellm import exceptions as litellm_exceptions
from pydantic import BaseModel

from llm_kit.config import resolve_model, validate_llm_config
from llm_kit.exceptions import LLMError, LLMOverloadedError, LLMRateLimitError
from llm_kit.pricing import compute_call_cost_usd
from llm_kit.telemetry import CallRecord, _fire_usage_callback, get_call_context

logger = logging.getLogger(__name__)

_DEFAULT_WAIT_S = 10.0
_MAX_WAIT_S = 60.0
_MAX_RETRIES = 3

# Read timeout for the local Ollama endpoint. A 30B MoE on CPU cold-loads
# several GB on every call when ``OLLAMA_KEEP_ALIVE=0`` and then does prefill +
# generation on the CPU — 120s is far too tight and surfaces as ReadTimeout.
# Generous default (15 min) to cover cold-load + a large-prompt bilingual
# generation under memory pressure + a payoff retry; override with
# ``OLLAMA_REQUEST_TIMEOUT_SECONDS``. Only affects the Ollama path; the
# Anthropic/Gemini paths use LiteLLM's own timeouts.
_OLLAMA_READ_TIMEOUT_S = float(os.environ.get("OLLAMA_REQUEST_TIMEOUT_SECONDS", "900"))


class LLMUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    web_searches: int = 0


@dataclass
class LLMResponse:
    """Normalized response shape across providers.

    ``text`` is the concatenated text content. ``raw`` holds the LiteLLM
    ``ModelResponse`` for callers that need provider-specific data
    (e.g. extended-thinking blocks under ``provider_specific_fields``).
    """

    text: str
    model: str
    usage: LLMUsage
    raw: Any = None


def _parse_retry_after(exc: Exception) -> float:
    """Honor server-provided ``retry-after`` (clamped to ``_MAX_WAIT_S``).

    LiteLLM rebuilds the upstream response into an ``httpx.Response``;
    ``.headers`` is a mapping-like ``httpx.Headers`` (NOT a ``dict``). We
    therefore call ``.get()`` directly without an ``isinstance(dict)`` check.
    """
    response: Any = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers is not None:
        try:
            raw = headers.get("retry-after")
        except Exception:
            raw = None
        if raw:
            try:
                return max(1.0, min(_MAX_WAIT_S, float(raw)))
            except (TypeError, ValueError):
                pass
    return _DEFAULT_WAIT_S


# Names of aiohttp connector exceptions we care about. Matched by class name
# instead of a hard aiohttp import because llm-kit doesn't depend on aiohttp
# directly — LiteLLM's Anthropic path pulls it in and raises these types when
# the TCP handshake fails or the peer drops mid-response.
_AIOHTTP_CONNECT_EXC_NAMES = frozenset(
    {
        "ClientConnectorError",
        "ClientConnectorSSLError",
        "ClientConnectorCertificateError",
        "ClientOSError",
        "ClientPayloadError",
        "ServerDisconnectedError",
        "ServerConnectionError",
    }
)


def _is_connect_error(exc: BaseException) -> bool:
    """True if ``exc`` (or anything in its cause chain) is a network-layer error.

    LiteLLM wraps TCP/SSL/DNS failures from httpx or aiohttp into a bare
    ``litellm.InternalServerError`` **without** ``status_code`` — the old
    heuristic (``status_code in {500, 529}``) missed them and every ideation
    call hard-failed on the first attempt during api.anthropic.com outages.
    Walking the ``__cause__`` / ``__context__`` chain lets us classify those
    as transient regardless of the outer wrapper.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(
            cur,
            (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
                ConnectionError,
                TimeoutError,
            ),
        ):
            return True
        if type(cur).__name__ in _AIOHTTP_CONNECT_EXC_NAMES:
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, litellm_exceptions.RateLimitError):
        return True
    if isinstance(exc, litellm_exceptions.ServiceUnavailableError):
        return True
    if isinstance(exc, litellm_exceptions.InternalServerError):
        # Anthropic's 529 "overloaded" surfaces as InternalServerError on
        # LiteLLM's side. We treat it transient like the SDK wrapper did.
        if getattr(exc, "status_code", None) in (500, 529):
            return True
        # LiteLLM also wraps httpx/aiohttp connect+read failures into a
        # status-code-less InternalServerError. Look through the cause chain
        # so a TCP outage triggers backoff instead of hard-failing on the
        # first attempt (see the 2026-07-27 api.anthropic.com blackout).
        if _is_connect_error(exc):
            return True
    return False


class LLMClient:
    """Provider-agnostic completion client backed by LiteLLM.

    Configure once at construction time. ``settings_obj`` (duck-typed) takes
    precedence; explicit kwargs override individual fields on top of it.
    """

    def __init__(
        self,
        settings_obj: Any | None = None,
        *,
        anthropic_api_key: str | None = None,
        gemini_api_key: str | None = None,
        ollama_base_url: str | None = None,
        ollama_model: str | None = None,
        llm_model: str | None = None,
        llm_provider: str | None = None,
    ) -> None:
        # Coalesce settings sources into a SimpleNamespace that resolve_model
        # / validate_llm_config can read uniformly. Explicit kwargs win.
        base = settings_obj if settings_obj is not None else SimpleNamespace()
        self._settings = SimpleNamespace(
            llm_provider=llm_provider or getattr(base, "llm_provider", None) or "anthropic",
            anthropic_api_key=anthropic_api_key
            if anthropic_api_key is not None
            else getattr(base, "anthropic_api_key", "") or "",
            gemini_api_key=gemini_api_key
            if gemini_api_key is not None
            else getattr(base, "gemini_api_key", "") or "",
            ollama_base_url=ollama_base_url
            if ollama_base_url is not None
            else getattr(base, "ollama_base_url", "") or "",
            ollama_model=ollama_model if ollama_model is not None else getattr(base, "ollama_model", "") or "",
            llm_model=llm_model if llm_model is not None else getattr(base, "llm_model", "") or "",
        )
        self._configured = False

    @classmethod
    def from_env(cls) -> LLMClient:
        """Build a client from the standard env vars.

        Reads ``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY``, ``LLM_PROVIDER``
        (default ``anthropic``), ``OLLAMA_BASE_URL``, ``OLLAMA_MODEL``,
        ``LLM_MODEL``. Does NOT validate — call
        :func:`llm_kit.validate_llm_config` against ``client._settings`` at
        app startup to fail fast on missing config.
        """
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", ""),
            ollama_model=os.environ.get("OLLAMA_MODEL", ""),
            llm_model=os.environ.get("LLM_MODEL", ""),
            llm_provider=os.environ.get("LLM_PROVIDER", "anthropic"),
        )

    def _ensure_configured(self) -> None:
        """One-time provider-specific setup. LiteLLM picks up provider keys
        from the environment, but we forward the configured value so a key
        loaded from .env via the consumer's settings works without exporting
        it explicitly."""
        if self._configured:
            return
        if self._settings.llm_provider == "anthropic":
            key = self._settings.anthropic_api_key or ""
            if key:
                litellm.anthropic_key = key
        elif self._settings.llm_provider == "gemini":
            key = self._settings.gemini_api_key or ""
            if key:
                # LiteLLM reads `gemini_api_key` for the `gemini/…` route, but
                # also accepts the canonical env var. Set both — env wins
                # over module attr in some LiteLLM versions.
                litellm.gemini_key = key
                os.environ.setdefault("GEMINI_API_KEY", key)
        self._configured = True

    def _record_usage(self, *, caller: str, model: str, usage: LLMUsage) -> None:
        """Price + dispatch one successful LLM call to the telemetry layer.

        Builds a :class:`CallRecord` carrying both the per-call usage and the
        current call-attribution context, then hands it off. The model string
        is forwarded *without* any provider prefix so :mod:`llm_kit.pricing`
        matches it via prefix lookup.
        """
        ctx = get_call_context()
        payload: dict[str, Any] = {
            "model": model,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": usage.cache_read_tokens,
            "cache_write_tokens": usage.cache_write_tokens,
            "web_searches": usage.web_searches,
        }
        cost_usd = compute_call_cost_usd(model, payload)
        record = CallRecord(
            caller=caller,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            web_searches=usage.web_searches,
            cost_usd=cost_usd,
            video_id=ctx["video_id"],
            channel_slug=ctx["channel_slug"],
            feature_prefix=ctx["feature_prefix"],
        )
        _fire_usage_callback(record)

    async def complete(
        self,
        *,
        caller: str,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | str | None = None,
        max_tokens: int = 1024,
        thinking_budget: int = 0,
        model: str | None = None,
        max_retries: int = _MAX_RETRIES,
        tools: list[dict[str, Any]] | None = None,
        provider: str | None = None,
        json_schema: dict[str, Any] | None = None,
        json_mode: bool = False,
        think: bool = False,
    ) -> LLMResponse:
        """Run a chat-style completion against the configured provider.

        Args:
            caller: Logical name for telemetry; flows into the CallRecord.
            messages: Chat messages, e.g. ``[{"role": "user", "content": "..."}]``.
                Multi-turn histories are accepted.
            system: Either a plain string (treated as a single uncached block)
                or a list of ``{"text": ..., "cache": "ephemeral" | None}`` blocks.
                Order is preserved; cache flags map to ``cache_control``.
            max_tokens: Output cap. When ``thinking_budget > 0``, Anthropic
                requires ``max_tokens > thinking_budget``; the caller owns that
                math so existing budget-aware logic is unchanged.
            thinking_budget: Extended-thinking budget in tokens (0 disables).
            model: Optional override. Defaults to the configured model for the
                active provider — provider prefix (``anthropic/...``) is added
                internally for Anthropic.
            tools: Optional Anthropic-format tool definitions, e.g.
                ``[{"type": "web_search_20250305", "name": "web_search",
                "max_uses": 2}]``. Forwarded verbatim to LiteLLM. Tool results
                surface under ``response.raw.choices[0].message.provider_specific_fields``;
                use :func:`extract_anthropic_tool_citations` to read them.
            provider: Per-call provider override (``"anthropic"`` | ``"ollama"``).
                When unset, falls back to the configured ``llm_provider``.
            json_schema: JSON schema enforced server-side (Ollama only) for
                structured output. Anthropic ignores this — call sites that
                need JSON over Anthropic continue to coerce via prompt + parse.

        Returns:
            :class:`LLMResponse` with concatenated text, usage breakdown
            (including cache tokens; cache fields are 0 for Ollama), and the
            raw provider response.
        """
        active_provider = provider or self._settings.llm_provider
        if active_provider not in {"anthropic", "ollama", "gemini"}:
            # Whitelist is enforced at boot via validate_llm_config; this is
            # a defense-in-depth guard for tests that bypass that path.
            validate_llm_config(self._settings)

        resolved_model = model or resolve_model(self._settings, provider=active_provider)

        if active_provider == "ollama":
            return await self._ollama_complete(
                caller=caller,
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                model=resolved_model,
                max_retries=max_retries,
                # Ollama: prefer caller-provided schema; if just json_mode and no
                # schema, fall back to format="json" (handled inside the method).
                json_schema=json_schema,
                json_mode=json_mode,
                think=think,
            )

        if active_provider == "gemini":
            return await self._gemini_complete(
                caller=caller,
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                model=resolved_model,
                max_retries=max_retries,
                json_mode=json_mode,
            )

        return await self._anthropic_complete(
            caller=caller,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            model=resolved_model,
            max_retries=max_retries,
            tools=tools,
            json_mode=json_mode,
        )

    async def _gemini_complete(
        self,
        *,
        caller: str,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | str | None,
        max_tokens: int,
        model: str,
        max_retries: int,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Gemini route via LiteLLM's ``gemini/<model>`` provider.

        Notes specific to Gemini (vs Anthropic):

        - System instruction goes **inside** the ``messages`` list as a
          ``{"role": "system", ...}`` turn. LiteLLM forwards it as
          ``systemInstruction`` to the Google API. Passing it as a top-
          level ``system=`` kwarg (like Anthropic) silently drops it,
          leaving the model with no instructions.
        - Gemini does NOT support Anthropic-style prefill (continuing
          from a partial assistant turn). If the caller appended an
          assistant turn for prefill purposes, drop it and rely on
          ``json_mode`` instead.
        - ``json_mode=True`` → ``response_format={"type":"json_object"}``
          (mapped to Gemini's ``responseMimeType: application/json`` by
          LiteLLM). This is the supported way to force JSON output.
        - Cache flags from the friendly system shape are dropped —
          Gemini's context caching has a different API.
        """
        self._ensure_configured()

        # Flatten the system blocks into a single string.
        system_text = self._to_ollama_system(system)

        # Strip any trailing assistant turn (Anthropic-style prefill) since
        # Gemini won't continue from it. The intent (force JSON) flows
        # through ``json_mode`` instead.
        gemini_messages: list[dict[str, Any]] = []
        if system_text:
            gemini_messages.append({"role": "system", "content": system_text})
        cleaned_user_messages = list(messages)
        while cleaned_user_messages and cleaned_user_messages[-1].get("role") == "assistant":
            cleaned_user_messages.pop()
        gemini_messages.extend(cleaned_user_messages)

        kwargs: dict[str, Any] = {
            "model": f"gemini/{model}",
            "max_tokens": max_tokens,
            "messages": gemini_messages,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(max_retries + 1):
            try:
                response = await litellm.acompletion(**kwargs)
                break
            except Exception as exc:
                if not _is_transient(exc) or attempt >= max_retries:
                    if isinstance(exc, litellm_exceptions.RateLimitError):
                        raise LLMRateLimitError(str(exc)) from exc
                    raise
                wait = _parse_retry_after(exc)
                logger.info(
                    "%s: transient gemini error (%s), sleeping %.1fs before retry (%d/%d)",
                    caller,
                    type(exc).__name__,
                    wait,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(wait)

        text = self._extract_text(response).strip()
        usage = self._extract_usage(response)
        self._record_usage(caller=caller, model=model, usage=usage)
        return LLMResponse(text=text, model=model, usage=usage, raw=response)

    async def _anthropic_complete(
        self,
        *,
        caller: str,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | str | None,
        max_tokens: int,
        thinking_budget: int,
        model: str,
        max_retries: int,
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        self._ensure_configured()

        anthropic_system = self._to_anthropic_system(system)
        # Anthropic-specific json forcing: prefill the assistant turn with "{"
        # so the model can only continue with valid JSON characters. We strip
        # any existing trailing assistant turn first to avoid double-prefill.
        effective_messages = list(messages)
        if json_mode:
            while effective_messages and effective_messages[-1].get("role") == "assistant":
                effective_messages.pop()
            effective_messages.append({"role": "assistant", "content": "{"})

        kwargs: dict[str, Any] = {
            "model": f"anthropic/{model}",
            "max_tokens": max_tokens,
            "messages": effective_messages,
        }
        if anthropic_system is not None:
            kwargs["system"] = anthropic_system
        if thinking_budget > 0:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        if tools:
            kwargs["tools"] = tools

        for attempt in range(max_retries + 1):
            try:
                response = await litellm.acompletion(**kwargs)
                break
            except Exception as exc:
                if not _is_transient(exc) or attempt >= max_retries:
                    if isinstance(exc, litellm_exceptions.RateLimitError):
                        raise LLMRateLimitError(str(exc)) from exc
                    if (
                        isinstance(exc, litellm_exceptions.InternalServerError)
                        and getattr(exc, "status_code", None) == 529
                    ):
                        raise LLMOverloadedError(str(exc)) from exc
                    raise
                wait = _parse_retry_after(exc)
                logger.info(
                    "%s: transient LLM error (%s), sleeping %.1fs before retry (%d/%d)",
                    caller,
                    type(exc).__name__,
                    wait,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(wait)

        text = self._extract_text(response).strip()
        # When json_mode pre-filled "{", the model response *continues* from
        # there; prepend the brace back so callers receive a complete JSON
        # object in resp.text.
        if json_mode:
            text = "{" + text
        usage = self._extract_usage(response)
        self._record_usage(caller=caller, model=model, usage=usage)
        return LLMResponse(text=text, model=model, usage=usage, raw=response)

    async def _ollama_complete(
        self,
        *,
        caller: str,
        messages: list[dict[str, Any]],
        system: list[dict[str, Any]] | str | None,
        max_tokens: int,
        model: str,
        max_retries: int,
        json_schema: dict[str, Any] | None,
        json_mode: bool = False,
        think: bool = False,
    ) -> LLMResponse:
        """Talk to a local Ollama server via the native ``/api/chat`` endpoint.

        We bypass LiteLLM here so we can pass a JSON schema in the ``format``
        field — Ollama enforces it server-side, which is the whole point of
        running this connector for structured-output areas. When
        ``json_schema`` is None the request falls back to ``format="json"`` so
        the response is at least guaranteed to parse as valid JSON.

        System blocks collapse to a single Ollama ``system`` message: the
        ``cache="ephemeral"`` flag is ignored (Ollama has no equivalent), but
        text order is preserved so prompt content stays identical to the
        Anthropic path.
        """
        base_url = (self._settings.ollama_base_url or "").rstrip("/")
        if not base_url:
            raise LLMError("OLLAMA_BASE_URL is not set; cannot route to local provider")

        ollama_messages: list[dict[str, Any]] = []
        system_text = self._to_ollama_system(system)
        if system_text:
            ollama_messages.append({"role": "system", "content": system_text})
        ollama_messages.extend(self._to_ollama_messages(messages))

        payload: dict[str, Any] = {
            "model": model,
            "messages": ollama_messages,
            "stream": False,
            # Thinking models (qwen3, deepseek-r1) put their answer in the
            # `thinking` field and leave `content` empty unless thinking is
            # disabled. We read `message.content`, so default think=False keeps
            # callers getting the answer; pass think=True to opt back in.
            "think": think,
            "options": {"num_predict": max_tokens},
        }
        payload["format"] = json_schema if json_schema is not None else "json"

        url = f"{base_url}/api/chat"
        last_exc: Exception | None = None
        async with httpx.AsyncClient(timeout=httpx.Timeout(_OLLAMA_READ_TIMEOUT_S, connect=10.0)) as client:
            for attempt in range(max_retries + 1):
                try:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    break
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    last_exc = exc
                    if status == 429:
                        if attempt >= max_retries:
                            raise LLMRateLimitError(str(exc)) from exc
                    elif 500 <= status < 600:
                        if attempt >= max_retries:
                            raise LLMError(f"ollama server error {status}: {exc}") from exc
                    else:
                        raise LLMError(f"ollama request failed ({status}): {exc.response.text[:200]}") from exc
                    wait = min(_MAX_WAIT_S, _DEFAULT_WAIT_S * (attempt + 1))
                    logger.info(
                        "%s: transient ollama error (status=%s), sleeping %.1fs (%d/%d)",
                        caller,
                        status,
                        wait,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(wait)
                except httpx.HTTPError as exc:
                    last_exc = exc
                    if attempt >= max_retries:
                        raise LLMError(f"ollama transport error: {exc}") from exc
                    wait = min(_MAX_WAIT_S, _DEFAULT_WAIT_S * (attempt + 1))
                    logger.info(
                        "%s: ollama transport error (%s), sleeping %.1fs (%d/%d)",
                        caller,
                        type(exc).__name__,
                        wait,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(wait)
            else:  # pragma: no cover — defensive; the loop always breaks or raises
                raise LLMError(f"ollama request exhausted retries: {last_exc}")

        data = response.json()
        message = data.get("message") or {}
        text = (message.get("content") or "").strip()
        usage = LLMUsage(
            input_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
        )
        self._record_usage(caller=caller, model=model, usage=usage)
        return LLMResponse(text=text, model=model, usage=usage, raw=data)

    @staticmethod
    def _to_ollama_system(system: list[dict[str, Any]] | str | None) -> str:
        """Flatten the friendlier system-block shape into a single string.

        Ollama's chat API takes one ``system`` message; the per-block
        ``cache="ephemeral"`` flag is meaningless locally and is silently
        dropped. Text order is preserved so the prompt content the model
        sees matches the Anthropic path verbatim.
        """
        if system is None:
            return ""
        if isinstance(system, str):
            return system
        parts: list[str] = []
        for entry in system:
            text = entry.get("text", "")
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    @staticmethod
    def _to_ollama_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Flatten each message's ``content`` to a plain string for Ollama.

        Ollama's ``/api/chat`` requires ``content`` to be a string, but callers
        may pass Anthropic-style content blocks (``[{"type": "text", "text":
        ...}]``, sometimes carrying ``cache_control``). We concatenate the text
        blocks in order — the same collapse :meth:`_to_ollama_system` does for
        the system prompt — so the prompt the local model sees matches the
        Anthropic path verbatim. Non-text blocks (e.g. images) are dropped:
        text models can't use them, and vision areas never route to Ollama.
        """
        flattened: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                parts = [block["text"] for block in content if isinstance(block, dict) and block.get("text")]
                msg = {**msg, "content": "\n\n".join(parts)}
            flattened.append(msg)
        return flattened

    @staticmethod
    def _to_anthropic_system(
        system: list[dict[str, Any]] | str | None,
    ) -> list[dict[str, Any]] | None:
        if system is None:
            return None
        if isinstance(system, str):
            return [{"type": "text", "text": system}]

        blocks: list[dict[str, Any]] = []
        for entry in system:
            text = entry.get("text", "")
            block: dict[str, Any] = {"type": "text", "text": text}
            cache = entry.get("cache")
            if cache == "ephemeral":
                block["cache_control"] = {"type": "ephemeral"}
            elif cache is not None:
                raise ValueError(f"unsupported cache mode {cache!r} — only 'ephemeral' or None are allowed")
            blocks.append(block)
        return blocks

    @staticmethod
    def _extract_text(response: Any) -> str:
        """LiteLLM normalizes Anthropic content into OpenAI-shaped choices.
        ``message.content`` is a string when there is no extended thinking,
        and a list of dicts otherwise. We concatenate every text-typed entry
        and skip ``thinking`` blocks (the caller can read those via ``raw``)."""
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                if block_type in (None, "text"):
                    text = block.get("text") if isinstance(block, dict) else getattr(block, "text", "")
                    if text:
                        parts.append(text)
            return "".join(parts)
        return ""

    @staticmethod
    def _extract_usage(response: Any) -> LLMUsage:
        usage_obj = getattr(response, "usage", None)
        if usage_obj is None:
            return LLMUsage()
        # IMPORTANT: LiteLLM's ``prompt_tokens`` for the Anthropic provider is
        # the SUM of the uncached input + cache creation + cache read, NOT the
        # uncached portion that ``anthropic.AsyncAnthropic`` returns under
        # ``usage.input_tokens``. Using it as-is would double-count cache
        # writes/reads against ``pricing.py`` (which charges them at a
        # different rate). LiteLLM exposes the breakdown under
        # ``prompt_tokens_details.text_tokens``; we prefer that and fall back
        # to a manual subtraction when the field is absent.
        cache_read = getattr(usage_obj, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage_obj, "cache_creation_input_tokens", 0) or 0
        prompt_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0

        details = getattr(usage_obj, "prompt_tokens_details", None)
        text_tokens = getattr(details, "text_tokens", None) if details is not None else None
        if isinstance(text_tokens, int):
            input_tokens = text_tokens
        else:
            input_tokens = max(0, prompt_tokens - cache_read - cache_write)

        # Anthropic surfaces web_search count under usage.server_tool_use.
        # LiteLLM forwards the field on the Anthropic provider — read it
        # defensively so a future LiteLLM version that drops the field just
        # falls back to 0 instead of crashing.
        web_searches = 0
        server_tool = getattr(usage_obj, "server_tool_use", None)
        if server_tool is not None:
            web_searches = getattr(server_tool, "web_search_requests", 0) or 0

        return LLMUsage(
            input_tokens=input_tokens,
            output_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            web_searches=web_searches,
        )


def extract_anthropic_tool_citations(response: LLMResponse) -> list[str]:
    """Pull every URL that LiteLLM surfaced from Anthropic's web_search tool.

    LiteLLM flattens the Anthropic response: the visible text is in
    ``response.text`` while the structured ``web_search_tool_result`` and
    ``citations`` live under
    ``response.raw.choices[0].message.provider_specific_fields``. This helper
    walks both buckets, deduplicates the URLs preserving insertion order,
    and returns a plain list.

    Returns an empty list when ``response.raw`` is None (e.g. unit-test fakes
    that don't populate it) or when the model answered without using the tool.
    """
    raw = response.raw
    if raw is None:
        return []
    choices = getattr(raw, "choices", None) or []
    seen: set[str] = set()
    urls: list[str] = []
    for choice in choices:
        message = getattr(choice, "message", None)
        if message is None:
            continue
        psf = getattr(message, "provider_specific_fields", None) or {}
        if not isinstance(psf, dict):
            continue
        for tool_result in psf.get("web_search_results") or []:
            inner = tool_result.get("content") if isinstance(tool_result, dict) else None
            for entry in inner or []:
                url = entry.get("url") if isinstance(entry, dict) else None
                if url and url not in seen:
                    seen.add(url)
                    urls.append(url)
        for citation_group in psf.get("citations") or []:
            iterable = citation_group if isinstance(citation_group, list) else [citation_group]
            for entry in iterable:
                url = entry.get("url") if isinstance(entry, dict) else None
                if url and url not in seen:
                    seen.add(url)
                    urls.append(url)
    return urls
