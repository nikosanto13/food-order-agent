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
