"""Provider-agnostic LLM interface + Mistral implementation.

The agent depends only on ``LLMClient``; messages and tool schemas use the
OpenAI/Mistral chat format, the lingua franca across providers. Swapping to
another provider means one new class here, nothing else. Tests use a scripted
fake that implements the same protocol.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import structlog

logger = structlog.get_logger(__name__)


class LLMError(RuntimeError):
    """The LLM call failed after the provider-level retries."""


@dataclass(frozen=True)
class ToolCallRequest:
    id: str
    name: str
    arguments: str  # raw JSON string as produced by the model: parse defensively


@dataclass(frozen=True)
class LLMResponse:
    content: str
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    prompt_tokens: int = 0  # includes cached_prompt_tokens
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    finish_reason: str | None = None
    latency_ms: float = 0.0


class LLMClient(Protocol):
    model: str

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse: ...


class MistralLLM:
    """Thin adapter over the official ``mistralai`` SDK.

    Transport-level resilience (429/5xx/connection errors) is delegated to the
    SDK's exponential-backoff retry config; anything that still fails surfaces as
    ``LLMError`` and the agent degrades gracefully.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "mistral-small-latest",
        temperature: float = 0.2,
        max_tokens: int = 600,
        timeout_s: float = 30,
    ) -> None:
        from mistralai.client import Mistral  # noqa: PLC0415 - keep import cost off test paths
        from mistralai.client.utils.retries import BackoffStrategy, RetryConfig  # noqa: PLC0415

        retry = RetryConfig(
            "backoff",
            BackoffStrategy(
                initial_interval=500, max_interval=8_000, exponent=2.0, max_elapsed_time=20_000
            ),
            retry_connection_errors=True,
        )
        self._client = Mistral(
            api_key=api_key, retry_config=retry, timeout_ms=int(timeout_s * 1000)
        )
        self.model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        started = time.perf_counter()
        try:
            res = self._client.chat.complete(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]  # TypedDict-compatible dicts
                tools=tools,  # type: ignore[arg-type]
                tool_choice="auto",
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:  # normalise every provider failure into LLMError
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        latency_ms = round((time.perf_counter() - started) * 1000, 1)

        if not res.choices:
            raise LLMError("LLM returned no choices")
        choice = res.choices[0]
        msg = choice.message
        tool_calls = [
            ToolCallRequest(
                id=tc.id or f"call_{i}",
                name=tc.function.name,
                arguments=tc.function.arguments
                if isinstance(tc.function.arguments, str)
                else json.dumps(tc.function.arguments),
            )
            for i, tc in enumerate(msg.tool_calls or [])
        ]
        return LLMResponse(
            content=self._content_to_text(msg.content),
            tool_calls=tool_calls,
            prompt_tokens=res.usage.prompt_tokens or 0,
            cached_prompt_tokens=self._cached_prompt_tokens(res.usage),
            completion_tokens=res.usage.completion_tokens or 0,
            model=res.model,
            finish_reason=str(choice.finish_reason) if choice.finish_reason else None,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        # Mistral may return a list of content chunks; keep only text chunks.
        parts = [
            getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else None)
            for c in content
        ]
        return "".join(p for p in parts if p)

    @staticmethod
    def _cached_prompt_tokens(usage: Any) -> int:
        """Read ``usage.prompt_tokens_details.cached_tokens``.

        The SDK's ``UsageInfo`` doesn't declare the field, so it arrives as an extra
        attribute holding either a dict or a model object depending on SDK version.
        """
        details = getattr(usage, "prompt_tokens_details", None)
        if details is None:
            extra = getattr(usage, "additional_properties", None) or {}
            details = extra.get("prompt_tokens_details")
        if details is None:
            return 0
        cached = (
            details.get("cached_tokens")
            if isinstance(details, dict)
            else getattr(details, "cached_tokens", None)
        )
        return int(cached or 0)
