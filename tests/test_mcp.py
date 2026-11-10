"""MCP contract tests against the local mock server (in-process and over HTTP)."""

import pytest

from food_agent.mcp_client import McpOrderSubmitter, SubmitOutcome
from food_agent.mock_server import create_server

EMAIL = "candidate@example.com"
FRIES = {"items": [{"item_id": "fries", "quantity": 1, "options": {"size": "large"}}]}


def http(url: str, **kwargs) -> McpOrderSubmitter:
    return McpOrderSubmitter(url, applicant_email=EMAIL, backoff_s=0, **kwargs)


def test_success_over_http(mock_server_url):
    res = http(mock_server_url).submit(FRIES)
    assert res.outcome is SubmitOutcome.SUCCESS
    assert res.result["order_id"].startswith("ORD-")
    assert res.result["total"] == 5.0
    assert res.attempts == 1


def test_over_limit_is_refused(mock_server_url):
    payload = {"items": [{"item_id": "margherita", "quantity": 4, "options": {"size": "large"}}]}
    res = http(mock_server_url, max_attempts=1).submit(payload)
    assert res.outcome is SubmitOutcome.REFUSED
    assert "limit" in res.result["error"]


def test_kitchen_busy_is_retried_then_reported(mock_server_url):
    res = http(mock_server_url, force_failure="kitchen_busy", max_attempts=3).submit(FRIES)
    assert res.outcome is SubmitOutcome.REFUSED
    assert res.attempts == 3
    assert res.result["success"] is False


def test_server_error_is_classified_from_is_error(mock_server_url):
    res = http(mock_server_url, force_failure="server_error", max_attempts=2).submit(FRIES)
    assert res.outcome is SubmitOutcome.SERVER_ERROR
    assert res.attempts == 2


def test_schema_violation_is_not_retried(mock_server_url):
    bad = {"items": [{"item_id": "fries", "quantity": 0}]}
    res = http(mock_server_url).submit(bad)
    assert res.outcome is SubmitOutcome.INVALID_REQUEST
    assert res.attempts == 1


def test_missing_applicant_email_is_rejected(mock_server_url):
    res = McpOrderSubmitter(mock_server_url, max_attempts=1).submit(FRIES)
    assert res.outcome is SubmitOutcome.INVALID_REQUEST
    assert "X-Applicant-Email" in res.result["error"]


def test_unreachable_server_is_safe_to_retry():
    sub = McpOrderSubmitter(
        "http://127.0.0.1:9/mcp", applicant_email=EMAIL, max_attempts=2, backoff_s=0, timeout_s=3
    )
    res = sub.submit(FRIES)
    assert res.outcome is SubmitOutcome.UNREACHABLE
    assert res.attempts == 2


def test_transient_failure_recovers_on_retry():
    # busy_rate=0.5 with a fixed seed: first draw busy, a later one succeeds.
    server = create_server(busy_rate=0.5, seed=3)
    res = McpOrderSubmitter(server, backoff_s=0, max_attempts=5).submit(FRIES)
    assert res.outcome is SubmitOutcome.SUCCESS
    assert res.attempts > 1


@pytest.mark.parametrize("outcome", list(SubmitOutcome))
def test_only_known_safe_outcomes_are_retryable(outcome):
    assert outcome.retryable == (
        outcome in {SubmitOutcome.REFUSED, SubmitOutcome.SERVER_ERROR, SubmitOutcome.UNREACHABLE}
    )


def test_null_special_instructions_are_rejected_like_prod():
    """Prod rejects ``special_instructions: null``; the mock must too, so we never send it."""
    res = McpOrderSubmitter(create_server(), backoff_s=0, max_attempts=1).submit(
        {**FRIES, "special_instructions": None}
    )
    assert res.result["success"] is False


def test_special_instructions_string_is_accepted():
    res = McpOrderSubmitter(create_server(), backoff_s=0).submit(
        {**FRIES, "special_instructions": "no salt"}
    )
    assert res.result["success"] is True
