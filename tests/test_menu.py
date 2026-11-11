from decimal import Decimal

import pytest
from pydantic import ValidationError

from food_agent.menu import Menu, load_menu


def test_loads_shipped_menu():
    menu = load_menu()
    assert menu.ids == [
        "classic_burger",
        "spicy_burger",
        "margherita",
        "fries",
        "onion_rings",
        "soda",
        "milkshake",
    ]
    burger = menu.get("classic_burger")
    assert burger is not None
    assert burger.base_price == Decimal("8.50")
    assert burger.options["size"].modifier("large") == Decimal("2.00")


def test_milkshake_flavor_has_no_default():
    shake = load_menu().get("milkshake")
    assert shake is not None
    flavor = shake.options["flavor"]
    assert flavor.required
    assert flavor.default is None


def test_prompt_rendering_flags_options_that_must_be_asked():
    text = load_menu().to_prompt()
    assert "milkshake" in text
    assert "NO DEFAULT: must ask" in text
    assert "large(+2.00)" in text


def _menu(option: dict) -> dict:
    return {"menu": [{"id": "x", "name": "X", "base_price": 1, "options": {"size": option}}]}


@pytest.mark.parametrize(
    ("option", "message"),
    [
        (
            {"type": "single_choice", "required": True, "choices": ["a"], "default": "b"},
            "default",
        ),
        (
            {
                "type": "single_choice",
                "required": True,
                "choices": ["a"],
                "price_modifier": {"zz": 1},
            },
            "unknown choices",
        ),
        ({"type": "single_choice", "required": True, "choices": ["a", "a"]}, "duplicate"),
    ],
)
def test_menu_integrity_errors_fail_at_load_time(option, message):
    with pytest.raises(ValidationError, match=message):
        Menu.model_validate(_menu(option))


def test_duplicate_item_ids_rejected():
    item = {"id": "x", "name": "X", "base_price": 1}
    with pytest.raises(ValidationError, match="duplicate menu item ids"):
        Menu.model_validate({"menu": [item, item]})
