import pytest

from food_agent.config import Settings


def test_public_import_path():
    from agent import FoodOrderAgent  # noqa: PLC0415 - the import itself is under test

    assert FoodOrderAgent.send.__name__ == "send"


def test_missing_api_key_fails_fast_with_a_clear_message(tmp_path):
    from agent import FoodOrderAgent  # noqa: PLC0415

    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
        FoodOrderAgent(settings=Settings(mistral_api_key=None, log_dir=tmp_path))
