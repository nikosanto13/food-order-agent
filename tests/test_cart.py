from decimal import Decimal

import pytest

from food_agent.cart import (
    Cart,
    OrderError,
    OrderPhase,
    OrderSession,
    SubmissionNotAllowedError,
    describe_line,
    sanitize_special_instructions,
)


# ------------------------------------------------------------------ pricing (brief examples)
def test_example_1_total(menu):
    cart = Cart(menu)
    line = cart.add("classic_burger", options={"size": "large"}, extras=["cheese", "bacon"])
    cart.update(line.line_id, remove_extras=["bacon"])
    assert cart.total == Decimal("11.50")
    assert describe_line(menu, cart.lines[0]) == "Large Classic Burger with cheese"


def test_example_2_total(menu):
    cart = Cart(menu)
    cart.add("margherita", options={"size": "medium"}, extras=["olives"])
    cart.add("fries", options={"size": "large"}, extras=["parmesan"])
    cola = cart.add("soda", options={"flavor": "cola"})
    cart.update(cola.line_id, options={"size": "large"})
    assert cart.total == Decimal("22.25")


def test_negative_modifier_and_quantity(menu):
    cart = Cart(menu)
    cart.add("margherita", quantity=2, options={"size": "small"})
    assert cart.total == Decimal("20.00")


# ------------------------------------------------------------------ validation
def test_defaults_are_resolved_explicitly(menu):
    line = Cart(menu).add("classic_burger")
    assert line.options == {"size": "regular", "patty": "beef"}


def test_tokens_are_normalized(menu):
    line = Cart(menu).add(" Spicy_Burger ", options={"Spice Level": "Extra Hot"})
    assert line.options["spice_level"] == "extra_hot"


def test_unicode_extra_matches(menu):
    line = Cart(menu).add("spicy_burger", extras=["jalapeños"])  # decomposed ñ
    assert line.extras == ("jalapeños",)


def test_duplicate_extras_are_deduped(menu):
    line = Cart(menu).add("fries", extras=["parmesan", "Parmesan"])
    assert line.extras == ("parmesan",)


def test_required_option_without_default_must_be_asked(menu):
    with pytest.raises(OrderError, match="requires a flavor"):
        Cart(menu).add("milkshake")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"item_id": "sushi"}, "Unknown menu item"),
        ({"item_id": "fries", "quantity": 0}, "Quantity"),
        ({"item_id": "fries", "quantity": 21}, "Quantity"),
        ({"item_id": "fries", "options": {"size": "xl"}}, "choices: small, medium, large"),
        ({"item_id": "fries", "options": {"crust": "thin"}}, "no option"),
        ({"item_id": "fries", "extras": ["bacon"]}, "valid extras: truffle_oil"),
        ({"item_id": "soda", "extras": ["ice"]}, "valid extras: none"),
    ],
)
def test_invalid_requests_are_rejected_with_actionable_errors(menu, kwargs, message):
    with pytest.raises(OrderError, match=message):
        Cart(menu).add(**kwargs)


def test_failed_update_leaves_line_untouched(menu):
    cart = Cart(menu)
    line = cart.add("fries")
    with pytest.raises(OrderError):
        cart.update(line.line_id, options={"size": "huge"})
    assert cart.get(line.line_id) == line


def test_remove_extra_that_is_not_present(menu):
    cart = Cart(menu)
    line = cart.add("fries", extras=["parmesan"])
    with pytest.raises(OrderError, match="does not have extra"):
        cart.update(line.line_id, remove_extras=["truffle_oil"])


def test_line_ids_are_stable_and_not_reused(menu):
    cart = Cart(menu)
    a = cart.add("fries")
    cart.remove(a.line_id)
    b = cart.add("fries")
    assert b.line_id != a.line_id
    with pytest.raises(OrderError, match="No line"):
        cart.get(a.line_id)


def test_submit_payload_shape(menu):
    cart = Cart(menu)
    cart.add("classic_burger", options={"size": "large"}, extras=["cheese"])
    assert cart.to_submit_payload() == {
        "items": [
            {
                "item_id": "classic_burger",
                "quantity": 1,
                "options": {"size": "large", "patty": "beef"},
                "extras": ["cheese"],
            }
        ],
    }


# ------------------------------------------------------------------ special instructions
def test_special_instructions_are_sanitized():
    assert sanitize_special_instructions("  no\x00onions\n\tplease\u202e ") == "no onions please"
    assert sanitize_special_instructions("   ") is None


def test_special_instructions_too_long_are_rejected_not_truncated():
    with pytest.raises(OrderError, match="limited to 200"):
        sanitize_special_instructions("a" * 201)


# ------------------------------------------------------------------ confirmation guard
def test_cannot_submit_without_review(menu):
    session = OrderSession(menu)
    session.add(item_id="fries")
    with pytest.raises(SubmissionNotAllowedError, match="not been reviewed"):
        session.authorize_submission(turn=1)


def test_cannot_submit_in_the_same_turn_as_review(menu):
    session = OrderSession(menu)
    session.add(item_id="fries")
    session.review(turn=1)
    with pytest.raises(SubmissionNotAllowedError, match="same turn"):
        session.authorize_submission(turn=1)
    assert session.authorize_submission(turn=2)["items"][0]["item_id"] == "fries"


def test_any_change_after_review_invalidates_it(menu):
    session = OrderSession(menu)
    line = session.add(item_id="fries")
    session.review(turn=1)
    session.update(line.line_id, quantity=2)
    assert session.phase is OrderPhase.BUILDING
    with pytest.raises(SubmissionNotAllowedError):
        session.authorize_submission(turn=2)


def test_review_rejects_empty_and_over_limit_orders(menu):
    session = OrderSession(menu)
    with pytest.raises(OrderError, match="empty"):
        session.review(turn=1)
    session.add(item_id="margherita", quantity=4, options={"size": "large"})  # 4 x 16 = 64
    assert session.cart.snapshot()["over_limit"] is True
    with pytest.raises(OrderError, match=r"exceeds the \$50.00 limit by \$14.00"):
        session.review(turn=1)


def test_mark_submitted_starts_a_fresh_cart(menu):
    session = OrderSession(menu)
    session.add(item_id="fries")
    session.review(turn=1)
    session.authorize_submission(turn=2)
    session.mark_submitted({"success": True, "order_id": "ORD-1"})
    assert session.cart.is_empty
    assert session.phase is OrderPhase.BUILDING
    assert session.submitted_orders[0]["result"]["order_id"] == "ORD-1"


# ------------------------------------------------------------------ partial removal
def test_remove_some_units_decrements_the_line(menu):
    cart = Cart(menu)
    line = cart.add("classic_burger", quantity=3, options={"size": "large"})
    removed, remaining = cart.remove(line.line_id, quantity=1)
    assert removed.quantity == 1
    assert remaining is not None
    assert remaining.quantity == 2
    assert remaining.line_id == line.line_id  # same line, still addressable
    assert cart.total == Decimal("21.00")


@pytest.mark.parametrize("quantity", [None, 3])
def test_remove_all_units_or_no_quantity_deletes_the_line(menu, quantity):
    cart = Cart(menu)
    line = cart.add("classic_burger", quantity=3)
    removed, remaining = cart.remove(line.line_id, quantity=quantity)
    assert removed.quantity == 3
    assert remaining is None
    assert cart.is_empty


def test_remove_more_units_than_present_is_rejected(menu):
    cart = Cart(menu)
    line = cart.add("fries", quantity=2)
    with pytest.raises(OrderError, match="only has 2; cannot remove 5"):
        cart.remove(line.line_id, quantity=5)
    assert cart.get(line.line_id).quantity == 2


def test_partial_removal_invalidates_review(menu):
    session = OrderSession(menu)
    line = session.add(item_id="fries", quantity=2)
    session.review(turn=1)
    session.remove(line.line_id, quantity=1)
    assert session.phase is OrderPhase.BUILDING


def test_checkout_requested_tracks_the_checkout_flow(menu):
    session = OrderSession(menu)
    session.add(item_id="margherita", quantity=4, options={"size": "large"})
    with pytest.raises(OrderError):
        session.review(turn=1)  # over the limit, but the customer did ask to check out
    assert session.checkout_requested is True
    session.clear()
    assert session.checkout_requested is False


# ------------------------------------------------------------------ partial update (split)
def test_update_some_units_splits_the_line(menu):
    cart = Cart(menu)
    line = cart.add("classic_burger", quantity=3, options={"size": "large"})
    changed = cart.update(
        line.line_id, units=1, add_extras=["cheese", "bacon", "avocado", "extra_patty"]
    )
    assert changed.line_id != line.line_id
    assert changed.quantity == 1
    assert cart.get(line.line_id).quantity == 2
    assert cart.get(line.line_id).extras == ()
    assert [describe_line(menu, ln) for ln in cart.lines] == [
        "2x Large Classic Burger",
        "Large Classic Burger with cheese, bacon, avocado and extra patty",
    ]
    assert cart.total == Decimal("39.00")  # 2 x 10.50 + 18.00


def test_update_with_units_equal_to_quantity_changes_the_whole_line(menu):
    cart = Cart(menu)
    line = cart.add("fries", quantity=2)
    changed = cart.update(line.line_id, units=2, options={"size": "large"})
    assert changed.line_id == line.line_id
    assert len(cart.lines) == 1
    assert changed.options["size"] == "large"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"units": 5, "add_extras": ["parmesan"]}, "units must be between 1 and 2"),
        ({"units": 1, "quantity": 3}, "either units"),
        ({"units": 1, "options": {"size": "huge"}}, "Invalid size"),
    ],
)
def test_invalid_split_leaves_the_cart_untouched(menu, kwargs, message):
    cart = Cart(menu)
    line = cart.add("fries", quantity=2)
    with pytest.raises(OrderError, match=message):
        cart.update(line.line_id, **kwargs)
    assert cart.lines == [line]
    assert cart.add("fries").line_id == line.line_id + 1  # no line id was consumed
