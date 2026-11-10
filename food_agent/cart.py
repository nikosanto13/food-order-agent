"""Cart domain: line validation, pricing, and the submission state machine.

Everything here is deterministic and LLM-free. The LLM proposes *intents*
(add this, change that); this module decides whether they are valid against the
menu and owns the resulting state. That split is what makes the agent testable
and safe: the model can be wrong, but it cannot put an invalid or unpriced item
into an order.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from food_agent.menu import Menu, MenuItem

ORDER_TOTAL_LIMIT = Decimal("50.00")
MAX_LINE_QUANTITY = 20
MAX_SPECIAL_INSTRUCTIONS_LEN = 200
_CENT = Decimal("0.01")


class OrderError(ValueError):
    """A request that conflicts with the menu or the order rules.

    The message is written to be shown to the LLM verbatim so it can correct
    itself or ask the user (e.g. listing the valid choices).
    """


class SubmissionNotAllowedError(OrderError):
    """Submission attempted without a fresh, user-acknowledged review."""


def normalize_token(value: str) -> str:
    """Canonicalize free-form tokens: ``"Extra Hot"`` -> ``"extra_hot"``.

    NFC normalization makes ``jalapeños`` match regardless of how the
    ``ñ`` was encoded.
    """
    v = unicodedata.normalize("NFC", value).strip().lower()
    return re.sub(r"[\s\-]+", "_", v)


def money(value: Decimal) -> str:
    return f"${value.quantize(_CENT):.2f}"


class LineItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    line_id: int
    item_id: str
    quantity: int
    options: dict[str, str]
    extras: tuple[str, ...] = ()

    def to_submit_item(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "quantity": self.quantity,
            "options": dict(self.options),
            "extras": list(self.extras),
        }


def _resolve_item(menu: Menu, item_id: str) -> MenuItem:
    item = menu.get(normalize_token(item_id))
    if item is None:
        raise OrderError(f"Unknown menu item {item_id!r}. Valid item ids: {', '.join(menu.ids)}.")
    return item


def build_line(
    menu: Menu,
    *,
    line_id: int,
    item_id: str,
    quantity: int,
    options: Mapping[str, str] | None = None,
    extras: Sequence[str] | None = None,
) -> LineItem:
    """Validate a requested line against the menu and return its canonical form.

    Defaults are resolved here so the cart (and the payload the kitchen gets) is
    always explicit, never "whatever the default happens to be".
    """
    item = _resolve_item(menu, item_id)

    if not 1 <= quantity <= MAX_LINE_QUANTITY:
        raise OrderError(f"Quantity must be between 1 and {MAX_LINE_QUANTITY}, got {quantity}.")

    requested = {normalize_token(k): normalize_token(str(v)) for k, v in (options or {}).items()}
    resolved: dict[str, str] = {}
    errors: list[str] = []

    unknown = sorted(set(requested) - set(item.options))
    if unknown:
        valid = ", ".join(item.options) or "none"
        errors.append(f"{item.name} has no option(s) {unknown}; valid options: {valid}.")

    missing: list[str] = []
    for name, spec in item.options.items():
        choice = requested.get(name)
        if choice is None:
            if spec.default is not None:
                resolved[name] = spec.default
            elif spec.required:
                missing.append(name)
            continue
        if choice not in spec.choices:
            errors.append(
                f"Invalid {name} {choice!r} for {item.name}; choices: {', '.join(spec.choices)}."
            )
            continue
        resolved[name] = choice

    for name in missing:
        choices = ", ".join(item.options[name].choices)
        errors.append(
            f"{item.name} requires a {name} and has no default; ask the customer ({choices})."
        )

    allowed_extras = item.extra_prices
    canonical_extras: list[str] = []
    for raw in extras or ():
        extra = normalize_token(raw)
        if extra not in allowed_extras:
            valid = ", ".join(allowed_extras) or "none"
            errors.append(f"{item.name} has no extra {raw!r}; valid extras: {valid}.")
        elif extra not in canonical_extras:  # dedupe, keep the customer's order
            canonical_extras.append(extra)

    if errors:
        raise OrderError(" ".join(errors))

    return LineItem(
        line_id=line_id,
        item_id=item.id,
        quantity=quantity,
        options=resolved,
        extras=tuple(canonical_extras),
    )


def sanitize_special_instructions(text: str | None) -> str | None:
    """Special instructions are untrusted user text forwarded to a third party.

    Strip control characters, collapse whitespace, and bound the length. Too-long
    input is rejected (not truncated) so nothing the customer said is silently lost.
    """
    if text is None:
        return None
    # Control/format characters (incl. newlines, zero-width, bidi overrides) become spaces.
    cleaned = "".join(" " if unicodedata.category(ch)[0] == "C" else ch for ch in text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    if len(cleaned) > MAX_SPECIAL_INSTRUCTIONS_LEN:
        raise OrderError(
            f"Special instructions are limited to {MAX_SPECIAL_INSTRUCTIONS_LEN} characters "
            f"(got {len(cleaned)}); ask the customer to shorten them."
        )
    return cleaned


def unit_price(menu: Menu, line: LineItem) -> Decimal:
    item = _resolve_item(menu, line.item_id)
    price = item.base_price
    for name, choice in line.options.items():
        price += item.options[name].modifier(choice)
    extra_prices = item.extra_prices
    price += sum((extra_prices[e] for e in line.extras), Decimal(0))
    return price


def line_total(menu: Menu, line: LineItem) -> Decimal:
    return unit_price(menu, line) * line.quantity


def describe_line(menu: Menu, line: LineItem) -> str:
    """Human-readable line, e.g. ``2x Large Classic Burger (chicken) with cheese and bacon``.

    Size leads; other options are shown only when they carry information
    (required, or different from the default).
    """
    item = _resolve_item(menu, line.item_id)
    parts: list[str] = []
    if line.quantity > 1:
        parts.append(f"{line.quantity}x")
    if "size" in line.options:
        parts.append(line.options["size"].capitalize())
    parts.append(item.name)
    shown = [
        choice.replace("_", " ")
        for name, choice in line.options.items()
        if name != "size" and (item.options[name].required or choice != item.options[name].default)
    ]
    text = " ".join(parts)
    if shown:
        text += f" ({', '.join(shown)})"
    if line.extras:
        names = [e.replace("_", " ") for e in line.extras]
        joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"
        text += f" with {joined}"
    return text


class Cart:
    """Mutable list of validated lines. Line ids are stable and never reused."""

    def __init__(self, menu: Menu) -> None:
        self._menu = menu
        self._lines: dict[int, LineItem] = {}
        self._next_id = 1
        self.special_instructions: str | None = None

    @property
    def lines(self) -> list[LineItem]:
        return list(self._lines.values())

    @property
    def is_empty(self) -> bool:
        return not self._lines

    def get(self, line_id: int) -> LineItem:
        try:
            return self._lines[line_id]
        except KeyError:
            existing = ", ".join(str(i) for i in self._lines) or "none (cart is empty)"
            raise OrderError(
                f"No line {line_id} in the order; existing lines: {existing}."
            ) from None

    def add(
        self,
        item_id: str,
        quantity: int = 1,
        options: Mapping[str, str] | None = None,
        extras: Sequence[str] | None = None,
    ) -> LineItem:
        line = build_line(
            self._menu,
            line_id=self._next_id,
            item_id=item_id,
            quantity=quantity,
            options=options,
            extras=extras,
        )
        self._lines[line.line_id] = line
        self._next_id += 1
        return line

    def update(
        self,
        line_id: int,
        *,
        quantity: int | None = None,
        options: Mapping[str, str] | None = None,
        add_extras: Sequence[str] | None = None,
        remove_extras: Sequence[str] | None = None,
        units: int | None = None,
    ) -> LineItem:
        """Apply a partial change and re-validate the *whole* line.

        ``units`` applies the change to only that many units of a multi-unit line
        ("three burgers, add extras to one of them"): the line is split, the
        original keeps ``quantity - units`` unchanged units, and a new line with
        ``units`` units carries the change. Returns the line that carries the
        change, so ``result.line_id != line_id`` means a split happened.
        """
        current = self.get(line_id)
        if units is not None and units != current.quantity:
            if quantity is not None:
                raise OrderError(
                    "Pass either units (change some of the line) or quantity (change how many), "
                    "not both."
                )
            if not 1 <= units < current.quantity:
                raise OrderError(
                    f"Line {line_id} has {current.quantity}; units must be between 1 and "
                    f"{current.quantity}. Omit units to change the whole line."
                )
            quantity = units
            split = True
        else:
            split = False
        merged_options = {**current.options, **(options or {})}
        drop = {normalize_token(e) for e in remove_extras or ()}
        not_present = sorted(drop - set(current.extras))
        if not_present:
            raise OrderError(
                f"Line {line_id} does not have extra(s) {not_present}; "
                f"current extras: {', '.join(current.extras) or 'none'}."
            )
        extras = [e for e in current.extras if e not in drop] + list(add_extras or ())
        line = build_line(  # validate before touching the cart: a failed change changes nothing
            self._menu,
            line_id=self._next_id if split else line_id,
            item_id=current.item_id,
            quantity=current.quantity if quantity is None else quantity,
            options=merged_options,
            extras=extras,
        )
        if split:
            self._lines[line_id] = current.model_copy(
                update={"quantity": current.quantity - line.quantity}
            )
            self._next_id += 1
        self._lines[line.line_id] = line
        return line

    def remove(self, line_id: int, quantity: int | None = None) -> tuple[LineItem, LineItem | None]:
        """Remove ``quantity`` units of a line, or the whole line when ``quantity`` is None.

        Returns ``(removed, remaining)``: ``removed`` describes what was taken out
        (with the removed quantity) and ``remaining`` is the line left behind, or
        None if the line is gone. "Remove one of the three burgers" is therefore a
        deterministic decrement, not a judgement call left to the model.
        """
        line = self.get(line_id)
        if quantity is None or quantity == line.quantity:
            del self._lines[line_id]
            return line, None
        if quantity < 1:
            raise OrderError(f"Quantity to remove must be at least 1, got {quantity}.")
        if quantity > line.quantity:
            raise OrderError(
                f"Line {line_id} only has {line.quantity}; cannot remove {quantity}. "
                "Omit quantity to remove the whole line."
            )
        remaining = line.model_copy(update={"quantity": line.quantity - quantity})
        self._lines[line_id] = remaining
        return line.model_copy(update={"quantity": quantity}), remaining

    @property
    def total(self) -> Decimal:
        return sum((line_total(self._menu, ln) for ln in self._lines.values()), Decimal(0))

    def to_submit_payload(self) -> dict[str, Any]:
        """The exact ``submit_order`` arguments. Built by code, never by the LLM."""
        return {
            "items": [ln.to_submit_item() for ln in self._lines.values()],
            "special_instructions": self.special_instructions,
        }

    def fingerprint(self) -> str:
        canonical = json.dumps(self.to_submit_payload(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def snapshot(self) -> dict[str, Any]:
        """Structured view for the LLM and for traces."""
        return {
            "lines": [
                {
                    "line_id": ln.line_id,
                    "description": describe_line(self._menu, ln),
                    "item_id": ln.item_id,
                    "quantity": ln.quantity,
                    "options": ln.options,
                    "extras": list(ln.extras),
                    "line_total": money(line_total(self._menu, ln)),
                }
                for ln in self._lines.values()
            ],
            "special_instructions": self.special_instructions,
            "total": money(self.total),
            "over_limit": self.total > ORDER_TOTAL_LIMIT,
        }

    def summary_text(self) -> str:
        if self.is_empty:
            return "Your order is empty."
        lines = [
            f"- {describe_line(self._menu, ln)}: {money(line_total(self._menu, ln))}"
            for ln in self._lines.values()
        ]
        if self.special_instructions:
            lines.append(f"Special instructions: {self.special_instructions}")
        lines.append(f"Total: {money(self.total)}")
        return "\n".join(lines)


class OrderPhase(StrEnum):
    BUILDING = "building"
    AWAITING_CONFIRMATION = "awaiting_confirmation"


class OrderSession:
    """Cart + confirmation guard.

    Invariant enforced in code (not in the prompt): an order can only be
    submitted if ``review()`` was called in an *earlier* turn and the cart has
    not changed since. The user has therefore seen exactly this cart and total,
    and replied, before anything leaves the process. Every mutation goes through
    this class so it can invalidate a pending review.
    """

    def __init__(self, menu: Menu, *, total_limit: Decimal = ORDER_TOTAL_LIMIT) -> None:
        self.menu = menu
        self.total_limit = total_limit
        self.cart = Cart(menu)
        self.phase = OrderPhase.BUILDING
        self._reviewed_fingerprint: str | None = None
        self._reviewed_turn: int | None = None
        # True once the customer has asked to check out (review attempted, even if it
        # failed, e.g. over the limit). While set, changes trigger a fresh review so the
        # agent never asks "Ready to submit?" about a cart it has not re-validated.
        self.checkout_requested = False
        self.submitted_orders: list[dict[str, Any]] = []

    def _invalidate_review(self) -> None:
        self.phase = OrderPhase.BUILDING
        self._reviewed_fingerprint = None
        self._reviewed_turn = None

    def add(self, **kwargs: Any) -> LineItem:
        line = self.cart.add(**kwargs)
        self._invalidate_review()
        return line

    def update(self, line_id: int, **kwargs: Any) -> LineItem:
        line = self.cart.update(line_id, **kwargs)
        self._invalidate_review()
        return line

    def remove(self, line_id: int, quantity: int | None = None) -> tuple[LineItem, LineItem | None]:
        removed, remaining = self.cart.remove(line_id, quantity)
        self._invalidate_review()
        return removed, remaining

    def clear(self) -> None:
        for line in self.cart.lines:
            self.cart.remove(line.line_id)
        self.checkout_requested = False  # starting over, not checking out
        self._invalidate_review()

    def set_special_instructions(self, text: str | None) -> str | None:
        self.cart.special_instructions = sanitize_special_instructions(text)
        self._invalidate_review()
        return self.cart.special_instructions

    def validate_for_submission(self) -> None:
        """Server-side-style validation of the whole order before it is offered or sent."""
        if self.cart.is_empty:
            raise OrderError("The order is empty; nothing to submit.")
        # Re-validate every line from scratch: defence in depth in case the menu
        # was reloaded or a line was constructed by a path other than build_line.
        for ln in self.cart.lines:
            build_line(
                self.menu,
                line_id=ln.line_id,
                item_id=ln.item_id,
                quantity=ln.quantity,
                options=ln.options,
                extras=ln.extras,
            )
        total = self.cart.total
        if total > self.total_limit:
            raise OrderError(
                f"The order total {money(total)} exceeds the {money(self.total_limit)} limit "
                f"by {money(total - self.total_limit)}. The customer must remove or change items."
            )

    def review(self, turn: int) -> str:
        self.checkout_requested = True
        self.validate_for_submission()
        self.phase = OrderPhase.AWAITING_CONFIRMATION
        self._reviewed_fingerprint = self.cart.fingerprint()
        self._reviewed_turn = turn
        return self.cart.summary_text()

    def authorize_submission(self, turn: int) -> dict[str, Any]:
        """Return the payload to submit, or raise if the guard is not satisfied."""
        if self.phase is not OrderPhase.AWAITING_CONFIRMATION or self._reviewed_turn is None:
            raise SubmissionNotAllowedError(
                "Cannot submit: the order has not been reviewed. Call review_order, show the "
                "summary to the customer and wait for their explicit confirmation."
            )
        if self._reviewed_turn >= turn:
            raise SubmissionNotAllowedError(
                "Cannot submit in the same turn as the review: the customer has not replied to "
                "the summary yet. Show it and wait for confirmation."
            )
        if self.cart.fingerprint() != self._reviewed_fingerprint:  # pragma: no cover - defensive
            self._invalidate_review()
            raise SubmissionNotAllowedError(
                "The order changed since it was reviewed; review again."
            )
        self.validate_for_submission()
        return self.cart.to_submit_payload()

    def mark_submitted(self, result: Mapping[str, Any]) -> None:
        self.submitted_orders.append(
            {"payload": self.cart.to_submit_payload(), "result": dict(result)}
        )
        self.cart = Cart(self.menu)
        self.checkout_requested = False
        self._invalidate_review()
