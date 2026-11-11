"""Mistral adapter: reading cached prompt tokens from the SDK's usage object.

The SDK does not declare ``prompt_tokens_details``, so it arrives in different shapes
depending on the SDK version. If this silently returned 0, every cache hit would be
billed at the full input rate in the traces.
"""

from types import SimpleNamespace

import pytest

from food_agent.llm import MistralLLM


@pytest.mark.parametrize(
    "usage",
    [
        SimpleNamespace(prompt_tokens_details={"cached_tokens": 12}),
        SimpleNamespace(prompt_tokens_details=SimpleNamespace(cached_tokens=12)),
        SimpleNamespace(additional_properties={"prompt_tokens_details": {"cached_tokens": 12}}),
    ],
)
def test_cached_tokens_read_from_sdk_usage(usage):
    assert MistralLLM._cached_prompt_tokens(usage) == 12


@pytest.mark.parametrize(
    "usage",
    [
        SimpleNamespace(),
        SimpleNamespace(additional_properties={}),
        SimpleNamespace(prompt_tokens_details={"cached_tokens": None}),
    ],
)
def test_missing_cached_tokens_means_zero(usage):
    assert MistralLLM._cached_prompt_tokens(usage) == 0


def test_real_sdk_usage_object():
    from mistralai.client.models import UsageInfo  # noqa: PLC0415

    usage = UsageInfo.model_validate(
        {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 64},
        }
    )
    assert MistralLLM._cached_prompt_tokens(usage) == 64
