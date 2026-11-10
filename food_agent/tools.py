"""The LLM's tool surface: schemas, argument validation, and execution.

Least privilege by construction:

* The model can only call the tools below. Each one maps to a single, narrow
  operation on ``OrderSession`` and every argument is validated by a Pydantic
  model before anything runs.
* ``submit_order`` takes **no arguments**. The payload sent to the MCP server is
  built from the validated cart by code, so the model cannot alter what gets
  submitted (no prompt injection can smuggle an item or price into the order).
  Submission is further gated by the confirmation state machine in
  ``OrderSession.authorize_submission``.
* The model never sees or holds the MCP client, URL or headers.

Every tool returns a JSON object. Failures are returned as data
(``{"ok": false, "error": ...}``) rather than raised, so the model can recover:
ask the customer, fix the arguments, or explain.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from food_agent.cart import MAX_LINE_QUANTITY, OrderError, OrderSession, describe_line
from food_agent.mcp_client import OrderSubmitter, SubmitOutcome

logger = structlog.get_logger(__name__)


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AddItemArgs(_Args):
    item_id: str = Field(description="Menu item id, exactly as listed in the menu.")
    quantity: int = Field(default=1, ge=1, le=MAX_LINE_QUANTITY)
    options: dict[str, str] = Field(
        default_factory=dict,
        description="Option name -> choice, e.g. {'size': 'large'}. Omit options to use defaults.",
    )
    extras: list[str] = Field(default_factory=list, description="Extra ids, e.g. ['cheese'].")


class UpdateItemArgs(_Args):
    line_id: int = Field(description="line_id of the order line to change.")
    quantity: int | None = Field(default=None, ge=1, le=MAX_LINE_QUANTITY)
    options: dict[str, str] | None = Field(
        default=None, description="Only the options to change, e.g. {'size': 'regular'}."
    )
    add_extras: list[str] | None = None
    remove_extras: list[str] | None = None
    units: int | None = Field(
        default=None,
        ge=1,
        le=MAX_LINE_QUANTITY,
        description=(
            "Apply the change to only this many units of a multi-unit line; the line is "
            "split. E.g. line '3x Large Classic Burger' + 'add cheese to one of them' -> "
            "units=1, add_extras=['cheese']. Omit to change every unit on the line."
        ),
    )


class RemoveItemArgs(_Args):
    line_id: int = Field(description="line_id of the order line to remove from.")
    quantity: int | None = Field(
        default=None,
        ge=1,
        le=MAX_LINE_QUANTITY,
        description=(
            "How many units to remove from the line. Omit to remove the whole line. "
            "E.g. line '3x Large Classic Burger' + 'remove one of them' -> quantity=1."
        ),
    )


class SpecialInstructionsArgs(_Args):
    text: str | None = Field(description="Instructions for the kitchen; null or '' clears them.")


class NoArgs(_Args):
    pass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args: type[_Args]

    def schema(self) -> dict[str, Any]:
        params = self.args.model_json_schema()
        params.pop("title", None)
        for prop in params.get("properties", {}).values():
            prop.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": params},
        }


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "add_item",
        "Add one line to the order. Call once per distinct item/configuration. Ask the customer "
        "first if a required option has no default.",
        AddItemArgs,
    ),
    ToolSpec(
        "update_item",
        "Change an existing line (quantity, options, extras). Use for 'make it large', "
        "'remove the bacon', 'two of those'. To change only SOME units of a multi-unit line "
        "('one of them with cheese'), pass units.",
        UpdateItemArgs,
    ),
    ToolSpec(
        "remove_item",
        "Remove items from the order. With quantity, removes only that many units of the "
        "line ('remove one of the large burgers'); without it, removes the whole line.",
        RemoveItemArgs,
    ),
    ToolSpec("clear_order", "Remove every line (customer wants to start over).", NoArgs),
    ToolSpec("view_order", "Return the current order with line ids and totals.", NoArgs),
    ToolSpec(
        "set_special_instructions",
        "Set free-text instructions for the kitchen (e.g. 'no onions').",
        SpecialInstructionsArgs,
    ),
    ToolSpec(
        "review_order",
        "Call when the customer is done ordering or asks to submit. Validates the order and "
        "returns the summary to present for confirmation. Does NOT submit.",
        NoArgs,
    ),
    ToolSpec(
        "submit_order",
        "Submit the reviewed order to the kitchen. Only call when the customer's latest message "
        "explicitly confirms the summary you presented. Takes no arguments.",
        NoArgs,
    ),
)

TOOL_SCHEMAS: list[dict[str, Any]] = [spec.schema() for spec in TOOL_SPECS]
_SPECS_BY_NAME = {spec.name: spec for spec in TOOL_SPECS}

_SUBMIT_GUIDANCE: dict[SubmitOutcome, str] = {
    SubmitOutcome.SUCCESS: "Order placed. Tell the customer the order id and estimated time.",
    SubmitOutcome.REFUSED: (
        "Order NOT placed (already retried automatically). Explain the server's error in plain "
        "words. The cart is kept: offer to try again or to change the order."
    ),
    SubmitOutcome.SERVER_ERROR: (
        "Order NOT placed: the ordering service had an internal error (already retried). "
        "Apologise and offer to try again shortly. The cart is kept."
    ),
    SubmitOutcome.INVALID_REQUEST: (
        "Order NOT placed due to a technical problem on our side. Apologise briefly; do not "
        "retry automatically. The cart is kept."
    ),
    SubmitOutcome.UNREACHABLE: (
        "Order NOT placed: the ordering service could not be reached. Offer to try again."
    ),
    SubmitOutcome.UNCERTAIN: (
        "It is UNKNOWN whether the order was placed (the connection dropped after sending). "
        "Tell the customer honestly; do not resubmit unless they explicitly ask."
    ),
}


@dataclass
class ToolResult:
    content: dict[str, Any]
    external_call: dict[str, Any] | None = None  # set when an MCP tool was actually invoked
    events: list[str] = field(default_factory=list)  # e.g. "reviewed", "submitted"
    # Changes the reply must state exactly (multi-unit lines, where "one of them" vs "all of
    # them" is easy to get wrong): line ids whose current description must appear, and
    # descriptions of multi-unit lines removed entirely.
    touched_lines: list[int] = field(default_factory=list)
    removed_lines: list[str] = field(default_factory=list)

    def to_message_content(self) -> str:
        return json.dumps(self.content, ensure_ascii=False, default=str)


class ToolExecutor:
    def __init__(self, session: OrderSession, submitter: OrderSubmitter) -> None:
        self._session = session
        self._submitter = submitter
        self._handlers: dict[str, Callable[[Any, int], ToolResult]] = {
            "add_item": self._add_item,
            "update_item": self._update_item,
            "remove_item": self._remove_item,
            "clear_order": self._clear_order,
            "view_order": self._view_order,
            "set_special_instructions": self._set_special_instructions,
            "review_order": self._review_order,
            "submit_order": self._submit_order,
        }

    def execute(self, name: str, raw_arguments: str, *, turn: int) -> ToolResult:
        spec = _SPECS_BY_NAME.get(name)
        if spec is None:
            return ToolResult({"ok": False, "error": f"Unknown tool {name!r}."})

        try:
            parsed = json.loads(raw_arguments or "{}")
            args = spec.args.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as exc:
            return ToolResult({"ok": False, "error": f"Invalid arguments for {name}: {exc}"})

        try:
            return self._handlers[name](args, turn)
        except OrderError as exc:
            return ToolResult({"ok": False, "error": str(exc)})

    # -- helpers
    def _order(self) -> dict[str, Any]:
        return self._session.cart.snapshot()

    def _line(self, line: Any) -> dict[str, Any]:
        return {"line_id": line.line_id, "description": describe_line(self._session.menu, line)}

    def _changed(
        self,
        content: dict[str, Any],
        turn: int,
        *,
        touched: list[int] | None = None,
        removed: list[str] | None = None,
    ) -> ToolResult:
        """Result of a successful cart change.

        If the customer is already checking out, re-run the review right away and
        attach it, so the model's next "Ready to submit?" is backed by a real,
        current review (brief example 2: "Make the cola large" -> new total ->
        "Ready to submit?"). If the cart still fails validation (e.g. over the limit)
        the error is attached instead and the order stays in the building phase.
        """
        content["order"] = self._order()
        result = ToolResult(content, touched_lines=touched or [], removed_lines=removed or [])
        if not self._session.checkout_requested:
            return result
        try:
            summary = self._session.review(turn)
        except OrderError as exc:
            content["review"] = {
                "ok": False,
                "error": str(exc),
                "next_step": "Tell the customer what still needs to change before submitting.",
            }
            return result
        content["review"] = {
            "ok": True,
            "summary": summary,
            "total": self._order()["total"],
            "next_step": "Confirm the change, present this summary and total, then ask the "
            "customer to confirm.",
        }
        result.events.append("reviewed")
        return result

    # -- handlers
    def _add_item(self, args: AddItemArgs, turn: int) -> ToolResult:
        line = self._session.add(
            item_id=args.item_id, quantity=args.quantity, options=args.options, extras=args.extras
        )
        return self._changed({"ok": True, "added": self._line(line)}, turn)

    def _update_item(self, args: UpdateItemArgs, turn: int) -> ToolResult:
        before = self._session.cart.get(args.line_id).quantity
        line = self._session.update(
            args.line_id,
            quantity=args.quantity,
            options=args.options,
            add_extras=args.add_extras,
            remove_extras=args.remove_extras,
            units=args.units,
        )
        content: dict[str, Any] = {"ok": True, "updated": self._line(line)}
        touched: list[int] = []
        if line.line_id != args.line_id:  # split: some units changed, the rest are untouched
            unchanged = self._session.cart.get(args.line_id)
            content["unchanged_on_original_line"] = self._line(unchanged)
            content["note"] = (
                f"Only {line.quantity} of {before} changed; the line was split. In the "
                "confirmation, mention both the changed and the unchanged items."
            )
            touched = [args.line_id, line.line_id]
        elif max(before, line.quantity) > 1:
            touched = [line.line_id]
        return self._changed(content, turn, touched=touched)

    def _remove_item(self, args: RemoveItemArgs, turn: int) -> ToolResult:
        removed, remaining = self._session.remove(args.line_id, args.quantity)
        content: dict[str, Any] = {
            "ok": True,
            "removed_quantity": removed.quantity,
            "removed": self._line(removed),
            "remaining_on_line": self._line(remaining) if remaining else None,
        }
        if remaining is not None:
            return self._changed(content, turn, touched=[remaining.line_id])
        removed_text = [content["removed"]["description"]] if removed.quantity > 1 else []
        return self._changed(content, turn, removed=removed_text)

    def _clear_order(self, _args: NoArgs, _turn: int) -> ToolResult:
        self._session.clear()
        return ToolResult({"ok": True, "order": self._order()})

    def _view_order(self, _args: NoArgs, _turn: int) -> ToolResult:
        return ToolResult({"ok": True, "order": self._order()})

    def _set_special_instructions(self, args: SpecialInstructionsArgs, turn: int) -> ToolResult:
        text = self._session.set_special_instructions(args.text)
        return self._changed({"ok": True, "special_instructions": text}, turn)

    def _review_order(self, _args: NoArgs, turn: int) -> ToolResult:
        summary = self._session.review(turn)
        return ToolResult(
            {
                "ok": True,
                "summary": summary,
                "total": self._order()["total"],
                "next_step": "Present this summary and total, then ask the customer to confirm.",
            },
            events=["reviewed"],
        )

    def _submit_order(self, _args: NoArgs, turn: int) -> ToolResult:
        payload = self._session.authorize_submission(turn)  # raises if guard not satisfied
        started = time.perf_counter()
        res = self._submitter.submit(payload)
        logger.info(
            "order.submit",
            outcome=res.outcome.value,
            attempts=res.attempts,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        external = {"name": "submit_order", "arguments": payload, "result": res.result}
        events = ["submit_attempted"]
        if res.ok:
            self._session.mark_submitted(res.result)
            events.append("submitted")
        return ToolResult(
            {
                "ok": res.ok,
                "outcome": res.outcome.value,
                "attempts": res.attempts,
                "server_result": res.result,
                "guidance": _SUBMIT_GUIDANCE[res.outcome],
            },
            external_call=external,
            events=events,
        )
