import pytest

from food_agent.config import Settings


def test_public_import_path():
    from agent import FoodOrderAgent  # noqa: PLC0415 - the import itself is under test

    assert FoodOrderAgent.send.__name__ == "send"


def test_missing_api_key_fails_fast_with_a_clear_message(tmp_path):
    from agent import FoodOrderAgent  # noqa: PLC0415

    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
        FoodOrderAgent(settings=Settings(mistral_api_key=None, log_dir=tmp_path))


def test_session_cost_summary_breaks_spend_down_by_source():
    from food_agent.observability import SessionUsage  # noqa: PLC0415
    from main import session_cost_summary  # noqa: PLC0415

    usage = SessionUsage()
    usage.add(
        {
            "llm_calls": 2,
            "prompt_tokens": 1200,
            "cached_prompt_tokens": 1000,
            "completion_tokens": 40,
            "llm_cost_usd": 0.0001,
        }
    )
    summary = session_cost_summary(usage)
    assert summary.splitlines() == [
        "Session cost (estimated, USD) over 1 turn:",
        "  llm           $0.000100  2 calls, 1,200 in (1,000 cached) / 40 out tokens",
        "  total         $0.000100",
    ]
