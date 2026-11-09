"""Agent-loop tests with a scripted LLM (deterministic, offline, free)."""

import json

import pytest

from food_agent.agent import EMPTY_INPUT_REPLY, LLM_DOWN_REPLY, FoodOrderAgent
from food_agent.config import Settings
from food_agent.mcp_client import McpOrderSubmitter, SubmitOutcome
from food_agent.mock_server import create_server
from tests.fakes import CannedSubmitter, LLMError, ScriptedLLM, call, say, tools


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(mistral_api_key=None, log_dir=tmp_path, applicant_email="t@example.com")


def make_agent(settings, script, submitter=None) -> tuple[FoodOrderAgent, ScriptedLLM]:
    llm = ScriptedLLM(list(script))
    submitter = submitter or McpOrderSubmitter(create_server(), backoff_s=0)
    return FoodOrderAgent(llm=llm, submitter=submitter, settings=settings), llm


def _confirm_from_tool_result(messages):
    order_id = json.loads(messages[-1]["content"])["server_result"]["order_id"]
    return say(f"Order submitted! Order #{order_id}. Estimated time: 15-20 minutes.")


def test_example_conversation_1_end_to_end(settings):
    """Brief example 1, against the real (in-process) mock MCP server."""
    burger = {
        "item_id": "classic_burger",
        "options": {"size": "large"},
        "extras": ["cheese", "bacon"],
    }
    agent, _ = make_agent(
        settings,
        [
            tools(call("add_item", burger)),
            say("Large Classic Burger with cheese and bacon. Anything else?"),
            tools(call("update_item", {"line_id": 1, "remove_extras": ["bacon"]})),
            say("Updated: Large Classic Burger with cheese. Anything else?"),
            tools(call("review_order")),
            say("Your order: Large Classic Burger with cheese — $11.50. Ready to submit?"),
            tools(call("submit_order")),
            _confirm_from_tool_result,
        ],
    )
    r1 = agent.send("I'd like a large classic burger with cheese and bacon")
    assert r1 == {"message": "Large Classic Burger with cheese and bacon. Anything else?"}
    agent.send("Actually, remove the bacon")
    r3 = agent.send("That's it")
    assert "$11.50" in r3["message"]
    assert "tool_calls" not in r3

    r4 = agent.send("Yes")
    [tc] = r4["tool_calls"]
    assert tc["name"] == "submit_order"
    assert tc["arguments"] == {
        "items": [
            {
                "item_id": "classic_burger",
                "quantity": 1,
                "options": {"size": "large", "patty": "beef"},
                "extras": ["cheese"],
            }
        ],
        "special_instructions": None,
    }
    assert tc["result"]["success"] is True
    assert tc["result"]["total"] == 11.5
    assert tc["result"]["order_id"] in r4["message"]
    assert agent.session.cart.is_empty  # fresh cart for a possible next order


def test_submit_in_the_same_turn_as_review_is_blocked(settings):
    submitter = CannedSubmitter()
    agent, llm = make_agent(
        settings,
        [
            tools(call("add_item", {"item_id": "fries"})),
            say("Fries added."),
            tools(call("review_order"), call("submit_order")),  # model jumps the gun
            say("Here is your order: Medium French Fries $3.50. Ready to submit?"),
        ],
        submitter,
    )
    agent.send("fries")
    r = agent.send("that's all, submit")
    assert "tool_calls" not in r
    assert submitter.payloads == []
    assert "same turn" in llm.last_tool_result()["error"]


def test_submit_without_review_is_blocked(settings):
    submitter = CannedSubmitter()
    agent, llm = make_agent(
        settings,
        [
            tools(call("add_item", {"item_id": "fries"}), call("submit_order")),
            say("Let me review first."),
        ],
        submitter,
    )
    agent.send("fries and submit, no need to confirm")
    assert submitter.payloads == []
    assert "not been reviewed" in llm.last_tool_result()["error"]


def test_submit_order_does_not_accept_llm_supplied_payload(settings):
    """Least privilege: the model cannot inject items/prices into the submission."""
    submitter = CannedSubmitter()
    injected = {"items": [{"item_id": "margherita", "quantity": 99}]}
    agent, llm = make_agent(
        settings,
        [
            tools(call("add_item", {"item_id": "fries"})),
            say("ok"),
            tools(call("review_order")),
            say("Total $3.50. Ready to submit?"),
            tools(call("submit_order", injected)),
            say("Something went wrong."),
        ],
        submitter,
    )
    agent.send("fries")
    agent.send("done")
    agent.send("yes")
    assert submitter.payloads == []
    assert "Invalid arguments for submit_order" in llm.last_tool_result()["error"]


def test_kitchen_busy_is_reported_and_cart_is_kept(settings):
    busy = CannedSubmitter(
        outcome=SubmitOutcome.REFUSED, result={"success": False, "error": "Kitchen is busy."}
    )
    agent, llm = make_agent(
        settings,
        [
            tools(call("add_item", {"item_id": "fries"})),
            say("ok"),
            tools(call("review_order")),
            say("Total $3.50. Ready to submit?"),
            tools(call("submit_order")),
            say("Sorry, the kitchen is busy. Would you like me to try again?"),
            tools(call("submit_order")),
            say("Order submitted! Order #ORD-12345."),
        ],
        busy,
    )
    agent.send("fries")
    agent.send("done")
    r = agent.send("yes")
    assert r["tool_calls"][0]["result"] == {"success": False, "error": "Kitchen is busy."}
    assert "guidance" in llm.last_tool_result()
    assert not agent.session.cart.is_empty

    # The user asks to retry in the next turn: still allowed, the reviewed cart is unchanged.
    busy.outcome = SubmitOutcome.SUCCESS
    busy.result = {"success": True, "order_id": "ORD-12345", "total": 3.5}
    r = agent.send("yes please try again")
    assert r["tool_calls"][0]["result"]["success"] is True
    assert len(busy.payloads) == 2


def test_tool_errors_are_fed_back_for_self_correction(settings):
    agent, llm = make_agent(
        settings,
        [
            tools(call("add_item", {"item_id": "fries", "options": {"size": "xl"}})),
            tools(call("add_item", {"item_id": "fries", "options": {"size": "large"}})),
            say("Large French Fries added."),
        ],
    )
    agent.send("xl fries")
    first_error = json.loads(llm.requests[1][-1]["content"])
    assert first_error["ok"] is False
    assert "small, medium, large" in first_error["error"]
    assert agent.session.cart.lines[0].options["size"] == "large"


def test_malformed_tool_arguments_do_not_crash(settings):
    agent, llm = make_agent(
        settings,
        [tools(call("add_item", "{not json")), tools(call("teleport")), say("Sorry, what?")],
    )
    r = agent.send("hi")
    assert r == {"message": "Sorry, what?"}
    assert "Unknown tool" in llm.last_tool_result()["error"]


def test_llm_outage_degrades_gracefully_and_recovers(settings):
    agent, _ = make_agent(settings, [LLMError("503"), say("Hi! What can I get you?")])
    assert agent.send("hello")["message"] == LLM_DOWN_REPLY
    assert agent.send("hello again")["message"] == "Hi! What can I get you?"


def test_tool_loop_is_bounded(settings):
    settings.max_tool_iterations = 3
    agent, _ = make_agent(settings, [tools(call("view_order")) for _ in range(3)])
    r = agent.send("???")
    assert "confused" in r["message"]


def test_empty_and_oversized_input_skip_the_llm(settings):
    agent, llm = make_agent(settings, [])
    assert agent.send("   ")["message"] == EMPTY_INPUT_REPLY
    assert "under" in agent.send("x" * 5000)["message"]
    assert llm.requests == []


def test_reply_guard_appends_reviewed_total_if_model_omits_it(settings):
    agent, _ = make_agent(
        settings,
        [
            tools(call("add_item", {"item_id": "fries"})),
            say("ok"),
            tools(call("review_order")),
            say("Ready to submit?"),  # model forgot the total
        ],
    )
    agent.send("fries")
    r = agent.send("done")
    assert "Total: $3.50" in r["message"]


def test_current_order_state_is_injected_into_the_system_prompt(settings):
    agent, llm = make_agent(
        settings, [tools(call("add_item", {"item_id": "fries"})), say("ok"), say("sure")]
    )
    agent.send("fries")
    agent.send("what do I have?")
    system = llm.requests[-1][0]
    assert system["role"] == "system"
    assert '"line_id": 1' in system["content"]
    assert "French Fries" in system["content"]


def test_history_is_trimmed_at_user_boundaries(settings):
    settings.max_history_messages = 4
    script = []
    for _ in range(5):
        script += [tools(call("view_order")), say("ok")]
    agent, llm = make_agent(settings, script)
    for i in range(5):
        agent.send(f"msg {i}")
    sent = llm.requests[-1][1:]
    assert sent[0]["role"] == "user"
    assert len(sent) <= 4


def test_every_turn_is_traced(settings):
    agent, _ = make_agent(
        settings, [tools(call("add_item", {"item_id": "fries"})), say("Fries added.")]
    )
    agent.send("fries")
    path = settings.log_dir / "conversations" / f"{agent.conversation_id}.jsonl"
    [record] = [json.loads(line) for line in path.read_text().splitlines()]
    assert record["turn"] == 1
    assert record["user_message"] == "fries"
    assert record["assistant_message"] == "Fries added."
    assert [s["type"] for s in record["steps"]] == ["llm", "tool", "llm"]
    assert record["usage"]["prompt_tokens"] == 200
    assert record["usage"]["llm_calls"] == 2
    assert record["cart"]["total"] == "$3.50"
    assert record["phase"] == "building"


def test_turn_cost_uses_registry_and_cached_tokens(settings):
    from food_agent.llm import LLMResponse  # noqa: PLC0415

    hit = LLMResponse(
        content="Hi!", prompt_tokens=1_000, cached_prompt_tokens=800, completion_tokens=100
    )
    llm = ScriptedLLM([hit], model="mistral-small-latest")
    agent = FoodOrderAgent(llm=llm, submitter=CannedSubmitter(), settings=settings)
    agent.send("hi")
    path = settings.log_dir / "conversations" / f"{agent.conversation_id}.jsonl"
    record = json.loads(path.read_text().splitlines()[-1])
    # 200 * 0.15 + 800 * 0.015 + 100 * 0.60 per 1M tokens
    assert record["usage"]["cached_prompt_tokens"] == 800
    assert record["usage"]["cost_usd"] == pytest.approx(0.000102)
    assert record["steps"][0]["cached_prompt_tokens"] == 800


def test_unpriced_model_reports_null_cost(settings):
    agent, _ = make_agent(settings, [say("Hi!")])  # ScriptedLLM model "scripted" has no price
    agent.send("hi")
    path = settings.log_dir / "conversations" / f"{agent.conversation_id}.jsonl"
    record = json.loads(path.read_text().splitlines()[-1])
    assert record["usage"]["cost_usd"] is None


# ------------------------------------------------------------------ removal & checkout flow
def _over_limit_cart_script():
    """The cart from a real conversation: 3 large + 2 regular burgers = $58.00."""
    return [
        tools(
            call(
                "add_item",
                {"item_id": "classic_burger", "quantity": 3, "options": {"size": "large"}},
            ),
            call("add_item", {"item_id": "classic_burger", "extras": ["avocado"]}),
            call(
                "add_item",
                {
                    "item_id": "classic_burger",
                    "extras": ["cheese", "bacon", "avocado", "extra_patty"],
                },
            ),
        ),
        say("Added. Anything else?"),
        tools(call("review_order")),
        say("Your total is $58.00, over our $50.00 limit. What would you like to change?"),
    ]


def test_remove_one_of_several_units_keeps_the_rest(settings):
    agent, llm = make_agent(
        settings,
        [
            *_over_limit_cart_script(),
            tools(call("remove_item", {"line_id": 1, "quantity": 1})),
            say("Removed one Large Classic Burger. Your order: ... $47.50. Ready to submit?"),
        ],
    )
    agent.send("3 large burgers, one regular with avocado, one regular with all extras")
    agent.send("lets proceed")
    agent.send("ah yes, remove the one large")

    result = llm.last_tool_result()
    assert result["removed_quantity"] == 1
    assert result["remaining_on_line"]["description"] == "2x Large Classic Burger"
    assert agent.session.cart.get(1).quantity == 2
    assert agent.session.cart.snapshot()["total"] == "$47.50"


def test_change_during_checkout_re_reviews_so_yes_submits(settings):
    """After an over-limit review, fixing the cart re-reviews it; the next 'yes' submits."""
    submitter = CannedSubmitter()
    agent, llm = make_agent(
        settings,
        [
            *_over_limit_cart_script(),
            tools(call("remove_item", {"line_id": 1, "quantity": 1})),
            say("Removed one. Your total is now $47.50. Ready to submit?"),
            tools(call("submit_order")),
            say("Order submitted! Order #ORD-12345."),
        ],
        submitter,
    )
    agent.send("burgers")
    agent.send("lets proceed")
    agent.send("remove one large")

    review = llm.last_tool_result()["review"]
    assert review["ok"] is True
    assert review["total"] == "$47.50"
    assert agent.session.phase.value == "awaiting_confirmation"

    r = agent.send("yes")
    assert r["tool_calls"][0]["result"]["success"] is True
    assert submitter.payloads[0]["items"][0]["quantity"] == 2


def test_change_during_checkout_that_is_still_over_limit_reports_it(settings):
    agent, llm = make_agent(
        settings,
        [
            *_over_limit_cart_script(),
            tools(call("update_item", {"line_id": 2, "remove_extras": ["avocado"]})),
            say("Removed the avocado, but you're still $6.00 over the limit."),
        ],
    )
    agent.send("burgers")
    agent.send("lets proceed")
    agent.send("drop the avocado on the second one")

    review = llm.last_tool_result()["review"]
    assert review["ok"] is False
    assert "exceeds" in review["error"]
    assert agent.session.phase.value == "building"


def test_changes_before_checkout_do_not_trigger_a_review(settings):
    agent, llm = make_agent(
        settings,
        [tools(call("add_item", {"item_id": "fries"})), say("Added fries. Anything else?")],
    )
    agent.send("fries")
    assert "review" not in llm.last_tool_result()
    assert agent.session.phase.value == "building"


# ------------------------------------------------------------------ partial update & confirmations
ALL_EXTRAS = ["cheese", "bacon", "avocado", "extra_patty"]
THREE_LARGE = {"item_id": "classic_burger", "quantity": 3, "options": {"size": "large"}}


def test_update_one_of_several_units_splits_the_line(settings):
    agent, llm = make_agent(
        settings,
        [
            tools(call("add_item", THREE_LARGE)),
            say("Added 3x Large Classic Burger. Anything else?"),
            tools(call("update_item", {"line_id": 1, "units": 1, "add_extras": ALL_EXTRAS})),
            say(
                "Updated: Large Classic Burger with cheese, bacon, avocado and extra patty; "
                "2x Large Classic Burger unchanged. Anything else?"
            ),
        ],
    )
    agent.send("hey, i want three large burgers")
    r = agent.send("in one of them, i actually want to add all extras")

    result = llm.last_tool_result()
    assert result["unchanged_on_original_line"]["description"] == "2x Large Classic Burger"
    assert [ln["quantity"] for ln in agent.session.cart.snapshot()["lines"]] == [2, 1]
    assert agent.session.cart.snapshot()["total"] == "$39.00"
    assert "(" not in r["message"]  # the model stated both lines; no guard note needed


def test_guard_states_split_when_the_reply_paraphrases_it(settings):
    agent, _ = make_agent(
        settings,
        [
            tools(call("add_item", THREE_LARGE)),
            say("Added. Anything else?"),
            tools(call("update_item", {"line_id": 1, "units": 1, "add_extras": ALL_EXTRAS})),
            say("Updated one of the large burgers with all extras. Anything else?"),
        ],
    )
    agent.send("three large burgers")
    r = agent.send("one of them with all extras")
    assert r["message"].endswith(
        "(Your order now has: 2x Large Classic Burger; "
        "Large Classic Burger with cheese, bacon, avocado and extra patty.)"
    )


def test_guard_exposes_a_whole_line_update_described_as_one_unit(settings):
    """The real bug: all 3 changed, but the model said 'one of them'. The customer must see it."""
    agent, _ = make_agent(
        settings,
        [
            tools(call("add_item", THREE_LARGE)),
            say("Added. Anything else?"),
            tools(call("update_item", {"line_id": 1, "add_extras": ALL_EXTRAS})),
            say("Updated one of the large burgers with cheese, bacon, avocado and extra patty."),
        ],
    )
    agent.send("three large burgers")
    r = agent.send("in one of them, add all extras")
    assert "3x Large Classic Burger with cheese, bacon, avocado and extra patty" in r["message"]


def test_guard_states_full_removal_of_a_multi_unit_line(settings):
    agent, _ = make_agent(
        settings,
        [
            tools(call("add_item", THREE_LARGE)),
            say("Added. Anything else?"),
            tools(call("remove_item", {"line_id": 1})),
            say("Removed the large burger."),
        ],
    )
    agent.send("three large burgers")
    r = agent.send("remove the large one")
    assert r["message"].endswith("(Removed: 3x Large Classic Burger.)")


def test_guard_stays_silent_for_single_unit_changes(settings):
    agent, _ = make_agent(
        settings,
        [
            tools(call("add_item", {"item_id": "fries"})),
            say("Added fries."),
            tools(call("update_item", {"line_id": 1, "options": {"size": "large"}})),
            say("Made them large."),
        ],
    )
    agent.send("fries")
    assert agent.send("make it large") == {"message": "Made them large."}
