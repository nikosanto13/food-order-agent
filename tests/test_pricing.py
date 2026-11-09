"""Cost estimation: per-model registry and prompt-cache-aware pricing."""

from decimal import Decimal
from types import SimpleNamespace

import pytest

from food_agent.llm import MistralLLM
from food_agent.pricing import (
    MODEL_PRICING,
    ModelPricing,
    estimate_cost,
    estimate_stt_cost,
    estimate_tts_cost,
)


def test_mistral_small_prices():
    p = MODEL_PRICING["mistral-small-latest"]
    assert (p.input, p.cached_input, p.output) == (
        Decimal("0.15"),
        Decimal("0.015"),
        Decimal("0.60"),
    )


def test_cached_tokens_are_billed_at_the_cached_rate():
    # 1M prompt tokens of which 400k cached, 100k output:
    # 600k * 0.15 + 400k * 0.015 + 100k * 0.60 = 0.09 + 0.006 + 0.06
    cost = estimate_cost(
        "mistral-small-latest",
        prompt_tokens=1_000_000,
        cached_prompt_tokens=400_000,
        completion_tokens=100_000,
    )
    assert cost == pytest.approx(0.156)


def test_no_cache_hit_bills_everything_at_the_input_rate():
    cost = estimate_cost(
        "mistral-small-latest", prompt_tokens=1_000_000, cached_prompt_tokens=0, completion_tokens=0
    )
    assert cost == pytest.approx(0.15)


@pytest.mark.parametrize(("cached", "expected"), [(-5, 0.15), (2_000_000, 0.015)])
def test_cached_count_is_clamped_to_the_prompt(cached, expected):
    p = ModelPricing(input=Decimal("0.15"), cached_input=Decimal("0.015"), output=Decimal("0.6"))
    assert p.cost(1_000_000, cached, 0) == pytest.approx(expected)


def test_unknown_model_has_no_estimate():
    assert (
        estimate_cost("nope", prompt_tokens=10, cached_prompt_tokens=0, completion_tokens=10)
        is None
    )


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


# ------------------------------------------------------------------ speech


def test_stt_is_billed_per_audio_hour_with_a_keyterms_add_on():
    assert estimate_stt_cost("scribe_v2", audio_s=3600, keyterms=False) == pytest.approx(0.22)
    assert estimate_stt_cost("scribe_v2", audio_s=3600, keyterms=True) == pytest.approx(0.27)
    assert estimate_stt_cost("scribe_v2", audio_s=2.5, keyterms=True) == pytest.approx(
        2.5 * 0.27 / 3600
    )


def test_tts_is_billed_per_thousand_characters():
    assert estimate_tts_cost("eleven_flash_v2_5", chars=1000) == pytest.approx(0.04)
    assert estimate_tts_cost("eleven_flash_v2_5", chars=250) == pytest.approx(0.01)


def test_unknown_speech_models_are_unpriced_not_free():
    assert estimate_stt_cost("scribe_v9", audio_s=10, keyterms=False) is None
    assert estimate_tts_cost("eleven_v9", chars=10) is None
