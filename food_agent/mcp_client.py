"""MCP client for ``submit_order`` with explicit failure classification and retries.

Design notes
------------
* **Connect per submission.** Submitting is rare (about once per conversation), so we
  open a Streamable HTTP session, call the tool and close it. That costs one
  handshake per order but keeps ``FoodOrderAgent.send()`` synchronous, avoids a
  long-lived background event loop, and makes per-request headers
  (``X-Demo-Force-Failure``) trivial.
* **Classify, then decide.** Every result maps to exactly one ``SubmitOutcome``.
  Retry policy keys off the outcome, never off error *text* (which is not part of
  the contract).
* **Idempotency.** We only retry when we *know* no order was placed: an explicit
  ``success: false`` body, or a failure *before* the tool call was sent. A
  timeout/disconnect *after* sending is ``UNCERTAIN``: retrying could
  double-order, so we surface it instead.
* **Refusals are presumed transient.** The agent validates the cart against the
  menu and the $50 limit before it ever calls the server, so a ``success: false``
  for a pre-validated cart is most likely the busy kitchen. Retrying a genuine
  business refusal unchanged is harmless (it is refused again), so we retry a
  bounded number of times and then let the LLM relay the server's message. If the
  server ever adds a machine-readable ``retryable`` flag, it is honoured.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

import anyio
import httpx2
import structlog
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.mcpserver import MCPServer
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp_types import CallToolResult, TextContent

log = structlog.get_logger(__name__)

TOOL_NAME = "submit_order"


class SubmitOutcome(StrEnum):
    SUCCESS = "success"
    REFUSED = "refused"  # success:false body (busy kitchen or business rule)
    SERVER_ERROR = "server_error"  # isError with a success:false body
    INVALID_REQUEST = "invalid_request"  # isError without body (schema, -32602): our bug
    UNREACHABLE = "unreachable"  # failed before the call was sent: safe to retry
    UNCERTAIN = "uncertain"  # failed after sending: order may exist, never auto-retry

    @property
    def retryable(self) -> bool:
        return self in {
            SubmitOutcome.REFUSED,
            SubmitOutcome.SERVER_ERROR,
            SubmitOutcome.UNREACHABLE,
        }


@dataclass(frozen=True)
class SubmitResult:
    outcome: SubmitOutcome
    result: dict[str, Any]  # what the server said (or a synthesized error body)
    attempts: int = 1
    attempt_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.outcome is SubmitOutcome.SUCCESS


class OrderSubmitter(Protocol):
    def submit(self, payload: dict[str, Any]) -> SubmitResult: ...


def _body_of(result: CallToolResult) -> dict[str, Any] | None:
    """Prefer ``structured_content``; fall back to JSON in the first text block."""
    if isinstance(result.structured_content, dict):
        return dict(result.structured_content)
    for block in result.content:
        if isinstance(block, TextContent):
            try:
                parsed = json.loads(block.text)
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, dict) else None
    return None


def _text_of(result: CallToolResult) -> str:
    return " ".join(b.text for b in result.content if isinstance(b, TextContent))[:500]


def classify(result: CallToolResult) -> tuple[SubmitOutcome, dict[str, Any]]:
    body = _body_of(result)
    if result.is_error:  # check isError *before* trusting structured content
        if body is not None and body.get("success") is False:
            return SubmitOutcome.SERVER_ERROR, body
        return SubmitOutcome.INVALID_REQUEST, {"success": False, "error": _text_of(result)}
    if body is None or "success" not in body:
        return SubmitOutcome.INVALID_REQUEST, {
            "success": False,
            "error": f"Unrecognised response: {_text_of(result)}",
        }
    if body["success"] is True:
        return SubmitOutcome.SUCCESS, body
    return SubmitOutcome.REFUSED, body


def _leaf_exceptions(exc: BaseException) -> list[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for sub in exc.exceptions for leaf in _leaf_exceptions(sub)]
    return [exc]


class McpOrderSubmitter:
    """Submits orders to an MCP server over Streamable HTTP (or in-process for tests)."""

    def __init__(
        self,
        server: str | MCPServer,
        *,
        applicant_email: str | None = None,
        timeout_s: float = 15.0,
        max_attempts: int = 3,
        backoff_s: float = 0.4,
        force_failure: str | None = None,
    ) -> None:
        self._server = server
        self._email = applicant_email
        self._timeout_s = timeout_s
        self._max_attempts = max_attempts
        self._backoff_s = backoff_s
        self.force_failure = force_failure  # X-Demo-Force-Failure, for failure drills

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._email:
            headers["X-Applicant-Email"] = self._email
        if self.force_failure:
            headers["X-Demo-Force-Failure"] = self.force_failure
        return headers

    def _client(self) -> Client:
        if isinstance(self._server, MCPServer):
            return Client(self._server)
        http = create_mcp_http_client(
            headers=self._headers(), timeout=httpx2.Timeout(self._timeout_s)
        )
        return Client(streamable_http_client(self._server, http_client=http))

    async def asubmit_once(self, payload: dict[str, Any]) -> tuple[SubmitOutcome, dict[str, Any]]:
        sent = False
        try:
            with anyio.fail_after(self._timeout_s):
                async with self._client() as client:
                    sent = True  # handshake done; from here a failure is ambiguous
                    result = await client.call_tool(TOOL_NAME, payload)
            return classify(result)
        except Exception as exc:  # noqa: BLE001 - every failure must map to an outcome
            leaves = _leaf_exceptions(exc)
            desc = "; ".join(f"{type(e).__name__}: {e}" for e in leaves)[:300]
            connect_failure = all(isinstance(e, httpx2.ConnectError) for e in leaves)
            outcome = (
                SubmitOutcome.UNREACHABLE
                if not sent or connect_failure
                else SubmitOutcome.UNCERTAIN
            )
            return outcome, {"success": False, "error": desc}

    def submit(self, payload: dict[str, Any]) -> SubmitResult:
        attempts: list[dict[str, Any]] = []
        outcome, body = SubmitOutcome.UNREACHABLE, {}
        for attempt in range(1, self._max_attempts + 1):
            started = time.perf_counter()
            outcome, body = anyio.run(self.asubmit_once, payload)
            latency_ms = round((time.perf_counter() - started) * 1000, 1)
            attempts.append(
                {"attempt": attempt, "outcome": outcome.value, "latency_ms": latency_ms}
            )
            log.info(
                "mcp.submit_attempt",
                attempt=attempt,
                outcome=outcome.value,
                latency_ms=latency_ms,
                error=body.get("error"),
            )
            retryable = outcome.retryable and body.get("retryable") is not False
            if not retryable or attempt == self._max_attempts:
                break
            # Exponential backoff with full jitter.
            time.sleep(random.uniform(0, self._backoff_s * 2 ** (attempt - 1)))  # noqa: S311
        return SubmitResult(
            outcome=outcome, result=body, attempts=len(attempts), attempt_log=attempts
        )
