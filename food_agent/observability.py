"""Structured logging and per-turn conversation traces.

Two complementary outputs:

* **Event log** (``logs/agent.jsonl``): one JSON object per event (LLM call, tool
  call, MCP attempt, turn end), every line carrying ``conversation_id`` and
  ``turn`` via contextvars, so ``jq 'select(.conversation_id=="...")'`` replays a
  conversation across processes.
* **Conversation trace** (``logs/conversations/<id>.jsonl``): exactly one record
  per turn with the user/assistant text, every step (LLM calls with latency and
  tokens, tool calls with arguments and results), the cart after the turn, the
  phase, and token/cost totals. This is the artefact you open to debug or eval a
  conversation, and it maps 1:1 onto an OpenTelemetry trace (turn = root span,
  steps = child spans) if this ever moves to Langfuse/Phoenix/OTel.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import structlog

_SENSITIVE_KEYS = {"api_key", "authorization", "mistral_api_key", "x-applicant-email", "password"}
_log_file: TextIO | None = None


def _redact(_: Any, __: str, event_dict: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    for key in list(event_dict):
        if key.lower() in _SENSITIVE_KEYS:
            event_dict[key] = "***"
    return event_dict


def configure_logging(
    *, level: str = "INFO", log_dir: Path = Path("logs"), to_stderr: bool = False
) -> None:
    """Configure structlog to emit JSON lines to a file (and optionally stderr).

    Logs go to a file by default so the interactive REPL stays clean; pass
    ``to_stderr=True`` (``main.py --verbose``) to watch them live.
    """
    global _log_file  # noqa: PLW0603 - process-wide logging sink
    log_dir.mkdir(parents=True, exist_ok=True)
    if _log_file is None:
        _log_file = (log_dir / "agent.jsonl").open("a", encoding="utf-8", buffering=1)
    sink: TextIO = sys.stderr if to_stderr else _log_file

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _redact,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(ensure_ascii=False, default=str),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelNamesMapping()[level]),
        logger_factory=structlog.WriteLoggerFactory(file=sink),
        cache_logger_on_first_use=False,
    )
    # Quiet chatty third-party stdlib loggers (HTTP clients, MCP internals).
    logging.basicConfig(level=logging.WARNING, stream=sink, format="%(name)s %(message)s")
    for name in ("httpx", "httpx2", "httpcore", "mcp", "uvicorn"):
        logging.getLogger(name).setLevel(logging.WARNING)


def new_conversation_id() -> str:
    return f"conv_{uuid.uuid4().hex[:12]}"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class TurnTrace:
    """Accumulates everything that happens in one ``send()`` call."""

    conversation_id: str
    turn: int
    user_message: str
    started_at: str = field(default_factory=_now)
    _t0: float = field(default_factory=time.perf_counter)
    steps: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)

    def add_step(self, kind: str, **data: Any) -> None:
        self.steps.append({"type": kind, "seq": len(self.steps) + 1, **data})

    def add_usage(self, prompt_tokens: int, completion_tokens: int, cost_usd: float) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cost_usd += cost_usd

    @property
    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 1)

    def to_record(
        self,
        *,
        assistant_message: str,
        external_tool_calls: list[dict[str, Any]],
        cart: Mapping[str, Any],
        phase: str,
    ) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "turn": self.turn,
            "started_at": self.started_at,
            "latency_ms": self.elapsed_ms,
            "user_message": self.user_message,
            "assistant_message": assistant_message,
            "steps": self.steps,
            "external_tool_calls": external_tool_calls,
            "cart": dict(cart),
            "phase": phase,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cost_usd": round(self.cost_usd, 6),
                "llm_calls": sum(1 for s in self.steps if s["type"] == "llm"),
            },
            "errors": self.errors,
        }


class ConversationTraceWriter:
    """Appends one JSON line per turn to ``<log_dir>/conversations/<conversation_id>.jsonl``."""

    def __init__(self, log_dir: Path) -> None:
        self._dir = log_dir / "conversations"

    def write(self, record: Mapping[str, Any]) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{record['conversation_id']}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return path
