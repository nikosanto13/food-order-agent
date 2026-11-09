"""Menu schema: parsing, integrity checks and lookups.

The YAML menu is the single source of truth for what can be ordered and what it
costs. It is parsed into immutable Pydantic models at startup, and structural
mistakes (a default that is not one of the choices, a price modifier for an
unknown choice, ...) fail fast at load time instead of surfacing mid-conversation.

Money is ``Decimal`` end-to-end: floats cannot represent 0.10 exactly, and a
food-ordering system that is off by a cent is a bug report waiting to happen.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_MENU_PATH = Path(__file__).resolve().parent.parent / "menu.yaml"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SingleChoiceOption(_Frozen):
    type: Literal["single_choice"]
    required: bool
    choices: tuple[str, ...] = Field(min_length=1)
    default: str | None = None
    price_modifier: dict[str, Decimal] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if len(set(self.choices)) != len(self.choices):
            raise ValueError(f"duplicate choices: {self.choices}")
        if self.default is not None and self.default not in self.choices:
            raise ValueError(f"default {self.default!r} is not one of {self.choices}")
        unknown = set(self.price_modifier) - set(self.choices)
        if unknown:
            raise ValueError(f"price_modifier for unknown choices: {sorted(unknown)}")
        return self

    def modifier(self, choice: str) -> Decimal:
        return self.price_modifier.get(choice, Decimal(0))


class Extra(_Frozen):
    id: str
    price: Decimal = Field(ge=0)


class ExtrasSpec(_Frozen):
    type: Literal["multi_choice"]
    choices: tuple[Extra, ...]

    @field_validator("choices")
    @classmethod
    def _unique_ids(cls, v: tuple[Extra, ...]) -> tuple[Extra, ...]:
        ids = [e.id for e in v]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate extra ids: {ids}")
        return v


class MenuItem(_Frozen):
    id: str
    name: str
    base_price: Decimal = Field(gt=0)
    options: dict[str, SingleChoiceOption] = Field(default_factory=dict)
    extras: ExtrasSpec | None = None

    @property
    def extra_prices(self) -> dict[str, Decimal]:
        return {e.id: e.price for e in self.extras.choices} if self.extras else {}

    def required_without_default(self) -> list[str]:
        """Options the customer *must* choose explicitly (e.g. milkshake flavor)."""
        return [n for n, o in self.options.items() if o.required and o.default is None]


class Menu(_Frozen):
    items: tuple[MenuItem, ...] = Field(alias="menu", min_length=1)

    @field_validator("items")
    @classmethod
    def _unique_ids(cls, v: tuple[MenuItem, ...]) -> tuple[MenuItem, ...]:
        ids = [i.id for i in v]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate menu item ids: {ids}")
        return v

    def get(self, item_id: str) -> MenuItem | None:
        return next((i for i in self.items if i.id == item_id), None)

    @property
    def ids(self) -> list[str]:
        return [i.id for i in self.items]

    def to_prompt(self) -> str:
        """Compact, token-cheap rendering of the menu for the system prompt.

        Rendered once per process and placed in the static prompt prefix so that
        provider-side prompt caching can reuse it across turns.
        """
        lines: list[str] = []
        for item in self.items:
            lines.append(f"- {item.id} ({item.name}): ${item.base_price:.2f}")
            for name, opt in item.options.items():
                rendered = []
                for c in opt.choices:
                    mod = opt.modifier(c)
                    rendered.append(f"{c}({mod:+.2f})" if mod else c)
                tags = ["required" if opt.required else "optional"]
                tags.append(f"default={opt.default}" if opt.default else "NO DEFAULT: must ask")
                lines.append(f"    {name} [{', '.join(tags)}]: {' | '.join(rendered)}")
            if item.extras:
                ex = ", ".join(f"{e.id}(+{e.price:.2f})" for e in item.extras.choices)
                lines.append(f"    extras: {ex}")
        return "\n".join(lines)


def load_menu(path: Path = DEFAULT_MENU_PATH) -> Menu:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)  # safe_load: never construct arbitrary Python objects
    return Menu.model_validate(raw)
