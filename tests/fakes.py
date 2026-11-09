"""Test doubles: a scripted LLM and a canned order submitter."""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from food_agent.llm import LLMError, LLMResponse, ToolCallRequest
from food_agent.mcp_client import SubmitOutcome, SubmitResult

_ids = itertools.count(1)


def call(name: str, arguments: dict[str, Any] | str | None = None) -> ToolCallRequest:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments or {})
    return ToolCallRequest(id=f"call{next(_ids):05d}", name=name, arguments=raw)


def tools(*calls: ToolCallRequest) -> LLMResponse:
    return LLMResponse(content="", tool_calls=list(calls), prompt_tokens=100, completion_tokens=20)


def say(text: str) -> LLMResponse:
    return LLMResponse(content=text, prompt_tokens=100, completion_tokens=10)


Step = LLMResponse | Exception | Callable[[list[dict[str, Any]]], LLMResponse]


@dataclass
class ScriptedLLM:
    """Returns pre-baked responses in order; records every request it receives."""

    script: list[Step]
    model: str = "scripted"
    requests: list[list[dict[str, Any]]] = field(default_factory=list)

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        self.requests.append([dict(m) for m in messages])
        if not self.script:
            raise AssertionError("ScriptedLLM ran out of responses")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        if callable(step) and not isinstance(step, LLMResponse):
            return step(messages)
        return step

    def last_tool_result(self) -> dict[str, Any]:
        tool_msgs = [m for m in self.requests[-1] if m["role"] == "tool"]
        return json.loads(tool_msgs[-1]["content"])


@dataclass
class CannedSubmitter:
    outcome: SubmitOutcome = SubmitOutcome.SUCCESS
    result: dict[str, Any] = field(
        default_factory=lambda: {
            "success": True,
            "order_id": "ORD-12345",
            "total": 11.5,
            "estimated_time": "15-20 minutes",
        }
    )
    payloads: list[dict[str, Any]] = field(default_factory=list)

    def submit(self, payload: dict[str, Any]) -> SubmitResult:
        self.payloads.append(payload)
        return SubmitResult(outcome=self.outcome, result=self.result, attempts=1)


__all__ = ["CannedSubmitter", "LLMError", "ScriptedLLM", "call", "say", "tools"]
