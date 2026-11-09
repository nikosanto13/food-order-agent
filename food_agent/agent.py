"""``FoodOrderAgent``: the conversational loop.

One ``send()`` call = one user turn:

1. Cheap pre-checks (empty / oversized input) answered without an LLM call.
2. Bounded tool-calling loop: LLM -> tool calls -> tool results -> LLM ... until the
   model answers in plain text or the step budget runs out.
3. Deterministic post-checks on the reply (the reviewed total and the order id
   must appear verbatim), so a model slip cannot misstate what the customer pays.
4. One trace record per turn, plus structured events along the way.

The cart (``OrderSession``) is the source of truth, not the chat history; the
current state is re-injected into the system prompt on every call. That lets
the history be trimmed aggressively (cost) without the model losing track of
the order (quality).
"""

from __future__ import annotations

from typing import Any

import structlog

from food_agent.cart import OrderError, OrderPhase, OrderSession, describe_line
from food_agent.config import Settings
from food_agent.llm import LLMClient, LLMError, MistralLLM
from food_agent.mcp_client import McpOrderSubmitter, OrderSubmitter
from food_agent.menu import Menu, load_menu
from food_agent.observability import (
    ConversationTraceWriter,
    SessionUsage,
    TraceSection,
    TurnTrace,
    configure_logging,
    merge_sections,
    new_conversation_id,
)
from food_agent.pricing import estimate_cost, pricing_for
from food_agent.prompts import build_static_prompt, build_system_message
from food_agent.tools import TOOL_SCHEMAS, ToolExecutor

logger = structlog.get_logger(__name__)

EMPTY_INPUT_REPLY = "Sorry, I didn't catch that. What would you like to order?"
LLM_DOWN_REPLY = "Sorry, I'm having trouble right now. Your order is saved; please try again in a moment."
STUCK_REPLY = "Sorry, I got a bit confused there. Here's your order so far:\n{summary}"


class FoodOrderAgent:
    """Conversational food-ordering agent.

    All collaborators are injectable (LLM, order submitter, menu, settings) so the
    agent runs offline in tests with a scripted LLM and the local MCP server.
    """

    def __init__(
        self,
        *,
        llm: LLMClient | None = None,
        submitter: OrderSubmitter | None = None,
        menu: Menu | None = None,
        settings: Settings | None = None,
        conversation_id: str | None = None,
    ) -> None:
        self.settings = settings or Settings()
        if not structlog.is_configured():
            configure_logging(
                level=self.settings.log_level,
                log_dir=self.settings.log_dir,
                to_stderr=self.settings.log_to_stderr,
            )
        self.menu = menu or load_menu()
        self.llm = llm or self._default_llm()
        if pricing_for(self.llm.model) is None:
            logger.warning(
                "llm.pricing_missing", model=self.llm.model, detail="cost_usd will be null"
            )
        self.submitter = submitter or McpOrderSubmitter(
            self.settings.mcp_url,
            applicant_email=self.settings.applicant_email,
            timeout_s=self.settings.mcp_timeout_s,
            max_attempts=self.settings.mcp_max_attempts,
            force_failure=self.settings.mcp_force_failure,
        )
        self._static_prompt = build_static_prompt(self.menu, self.settings.reply_style)
        self._traces = ConversationTraceWriter(self.settings.log_dir)
        self._open_record: dict[str, Any] | None = None
        self.usage = SessionUsage()  # whole session: survives reset()
        self._keep_turn_open = False
        self.reset(conversation_id)

    def _default_llm(self) -> LLMClient:
        key = self.settings.mistral_api_key
        if key is None:
            raise RuntimeError("MISTRAL_API_KEY is not set (see .env.example).")
        return MistralLLM(
            key.get_secret_value(),
            model=self.settings.mistral_model,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_output_tokens,
            timeout_s=self.settings.llm_timeout_s,
        )

    def reset(self, conversation_id: str | None = None) -> None:
        """Start a new conversation (fresh cart, history and conversation id)."""
        self.close_turn()
        self.conversation_id = conversation_id or new_conversation_id()
        self.session = OrderSession(self.menu)
        self._tools = ToolExecutor(self.session, self.submitter)
        self._history: list[dict[str, Any]] = []
        self._turn = 0

    def send(self, message: str, *, keep_turn_open: bool = False) -> dict[str, Any]:
        """Process one user message and return the reply (see the brief's format).

        The turn's trace record is written before returning, unless ``keep_turn_open``:
        then it is held until ``close_turn()``, so a front-end can add what happens
        around the agent (the voice channel adds transcription and speech timings and
        cost). A turn left open is closed by the next ``send()`` or ``reset()``.
        """
        self.close_turn()
        self._keep_turn_open = keep_turn_open
        self._turn += 1
        structlog.contextvars.bind_contextvars(
            conversation_id=self.conversation_id, turn=self._turn
        )
        try:
            return self._send(message)
        finally:
            structlog.contextvars.unbind_contextvars("conversation_id", "turn")

    def close_turn(self, *sections: TraceSection) -> None:
        """Write the open turn's trace record, with ``sections`` merged in. No-op if none."""
        record, self._open_record = self._open_record, None
        if record is None:
            return
        self.usage.add(record["usage"], sections)
        if sections:
            merge_sections(record, sections)
            logger.info(
                "turn.closed",
                conversation_id=record["conversation_id"],
                turn=record["turn"],
                **{k: v for k, v in record["usage"].items() if k.endswith("cost_usd")},
            )
        self._traces.write(record)

    def _send(self, message: str) -> dict[str, Any]:
        text = (message or "").strip()
        trace = TurnTrace(self.conversation_id, self._turn, text)
        logger.info("turn.start", chars=len(text))

        if not text:
            return self._finish(trace, EMPTY_INPUT_REPLY, [])
        limit = self.settings.max_user_message_chars
        if len(text) > limit:
            trace.errors.append("input_too_long")
            return self._finish(
                trace,
                f"That message is a bit long; could you keep it under {limit} characters?",
                [],
            )

        self._history.append({"role": "user", "content": text})
        external_calls: list[dict[str, Any]] = []
        events: set[str] = set()
        touched_lines: list[int] = []
        removed_lines: list[str] = []
        reply: str | None = None

        for _ in range(self.settings.max_tool_iterations):
            try:
                response = self._call_llm(trace)
            except LLMError as exc:
                trace.errors.append(f"llm_error: {exc}")
                logger.exception("llm.failed")
                reply = LLM_DOWN_REPLY
                break

            if not response.tool_calls:
                reply = response.content.strip() or "Anything else I can get you?"
                break

            self._history.append(
                {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                        for tc in response.tool_calls
                    ],
                }
            )
            for tc in response.tool_calls:
                result = self._tools.execute(tc.name, tc.arguments, turn=self._turn)
                events.update(result.events)
                touched_lines.extend(result.touched_lines)
                removed_lines.extend(result.removed_lines)
                if result.external_call is not None:
                    external_calls.append(result.external_call)
                trace.add_step(
                    "tool",
                    name=tc.name,
                    arguments=tc.arguments,
                    ok=result.content.get("ok"),
                    result=result.content,
                )
                logger.info("tool.call", tool=tc.name, ok=result.content.get("ok"))
                self._history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": result.to_message_content(),
                    }
                )
        else:
            trace.errors.append("max_tool_iterations")
            logger.warning("agent.max_iterations")
            reply = STUCK_REPLY.format(summary=self.session.cart.summary_text())

        assert reply is not None  # every branch above sets it  # noqa: S101
        reply = self._enforce_reply_invariants(
            reply, events, external_calls, touched_lines, removed_lines
        )
        self._history.append({"role": "assistant", "content": reply})
        return self._finish(trace, reply, external_calls)

    def _call_llm(self, trace: TurnTrace) -> Any:
        messages = [
            build_system_message(
                self._static_prompt, self.session.cart.snapshot(), self.session.phase.value
            ),
            *self._trimmed_history(),
        ]
        response = self.llm.complete(messages, TOOL_SCHEMAS)
        # Price by the model we asked for: the response may echo a dated snapshot id.
        cost = estimate_cost(
            self.llm.model,
            prompt_tokens=response.prompt_tokens,
            cached_prompt_tokens=response.cached_prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        trace.add_usage(
            response.prompt_tokens,
            response.cached_prompt_tokens,
            response.completion_tokens,
            cost,
        )
        trace.add_step(
            "llm",
            model=response.model or self.llm.model,
            latency_ms=response.latency_ms,
            prompt_tokens=response.prompt_tokens,
            cached_prompt_tokens=response.cached_prompt_tokens,
            completion_tokens=response.completion_tokens,
            cost_usd=cost,
            finish_reason=response.finish_reason,
            tool_calls=[tc.name for tc in response.tool_calls],
        )
        logger.info(
            "llm.call",
            model=response.model or self.llm.model,
            latency_ms=response.latency_ms,
            prompt_tokens=response.prompt_tokens,
            cached_prompt_tokens=response.cached_prompt_tokens,
            completion_tokens=response.completion_tokens,
            cost_usd=cost,
            tool_calls=[tc.name for tc in response.tool_calls],
        )
        return response

    def _trimmed_history(self) -> list[dict[str, Any]]:
        """Keep the last N messages, cut at a user-message boundary.

        Cutting elsewhere could orphan a tool result from its tool call, which
        providers reject. The cart lives in the system prompt, so old turns are
        expendable.
        """
        limit = self.settings.max_history_messages
        if len(self._history) <= limit:
            return self._history
        start = len(self._history) - limit
        while start < len(self._history) and self._history[start]["role"] != "user":
            start += 1
        return self._history[start:]

    def _enforce_reply_invariants(
        self,
        reply: str,
        events: set[str],
        external_calls: list[dict[str, Any]],
        touched_lines: list[int] | None = None,
        removed_lines: list[str] | None = None,
    ) -> str:
        """Deterministic guards on facts the customer must get exactly right."""
        awaiting = self.session.phase is OrderPhase.AWAITING_CONFIRMATION
        summary_shown = False
        if "reviewed" in events and awaiting and "submit_attempted" not in events:
            summary = self.session.cart.summary_text()
            total = self.session.cart.snapshot()["total"]
            if total not in reply:
                logger.warning("reply.guard", guard="review_total_missing")
                reply = f"{reply}\n\n{summary}\nReady to submit?"
                summary_shown = True
        if not summary_shown:
            reply = self._state_multi_unit_changes(reply, touched_lines or [], removed_lines or [])
        if "submitted" in events:
            order_id = str(external_calls[-1]["result"].get("order_id", ""))
            if order_id and order_id not in reply:
                logger.warning("reply.guard", guard="order_id_missing")
                reply = f"{reply}\nOrder #{order_id}."
        return reply

    def _state_multi_unit_changes(
        self, reply: str, touched_lines: list[int], removed_lines: list[str]
    ) -> str:
        """Changes to multi-unit lines must be stated exactly as they now are.

        "One of them" vs "all of them" is where a paraphrased confirmation goes wrong,
        e.g. "Updated one of the burgers" after all three were changed. If the reply
        does not contain the current description of each such line, append it. The
        check uses the cart *after* the turn, so later changes in the same turn win.
        """
        lowered = reply.lower()
        missing: list[str] = []
        for line_id in dict.fromkeys(touched_lines):  # dedupe, keep order
            try:
                line = self.session.cart.get(line_id)
            except OrderError:
                continue  # removed later in the same turn
            desc = describe_line(self.menu, line)
            if desc.lower() not in lowered:
                missing.append(desc)
        gone = [d for d in dict.fromkeys(removed_lines) if d.lower() not in lowered]
        if not missing and not gone:
            return reply
        logger.warning("reply.guard", guard="multi_unit_change_unstated")
        notes = []
        if gone:
            notes.append(f"Removed: {', '.join(gone)}.")
        if missing:
            notes.append(f"Your order now has: {'; '.join(missing)}.")
        return f"{reply}\n({' '.join(notes)})"

    def _finish(
        self, trace: TurnTrace, reply: str, external_calls: list[dict[str, Any]]
    ) -> dict[str, Any]:
        record = trace.to_record(
            assistant_message=reply,
            external_tool_calls=external_calls,
            cart=self.session.cart.snapshot(),
            phase=self.session.phase.value,
        )
        if self._keep_turn_open:
            self._open_record = record
        else:
            self.usage.add(record["usage"])
            self._traces.write(record)
        logger.info(
            "turn.end",
            latency_ms=record["latency_ms"],
            phase=record["phase"],
            **record["usage"],
            errors=trace.errors or None,
        )
        response: dict[str, Any] = {"message": reply}
        if external_calls:
            response["tool_calls"] = external_calls
        return response
