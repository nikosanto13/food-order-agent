"""Cost estimation: prompt-cache-aware LLM pricing, speech pricing, unpriced models.

Pricing objects are built here rather than read from the registries, so updating a
price in ``pricing.py`` never breaks a test.
"""

from decimal import Decimal

import pytest

from food_agent.pricing import (
    ModelPricing,
    SpeechToTextPricing,
    TextToSpeechPricing,
    estimate_cost,
    estimate_stt_cost,
    estimate_tts_cost,
)

LLM = ModelPricing(input=Decimal("0.15"), cached_input=Decimal("0.015"), output=Decimal("0.60"))


def test_cached_tokens_are_billed_at_the_cached_rate():
    # 1M prompt tokens of which 400k cached, 100k output:
    # 600k * 0.15 + 400k * 0.015 + 100k * 0.60 = 0.09 + 0.006 + 0.06
    assert LLM.cost(1_000_000, 400_000, 100_000) == pytest.approx(0.156)


@pytest.mark.parametrize(("cached", "expected"), [(-5, 0.15), (2_000_000, 0.015)])
def test_cached_count_is_clamped_to_the_prompt(cached, expected):
    assert LLM.cost(1_000_000, cached, 0) == pytest.approx(expected)


def test_speech_is_billed_per_audio_hour_and_per_thousand_characters():
    stt = SpeechToTextPricing(per_hour=Decimal("0.20"), keyterms_per_hour=Decimal("0.05"))
    assert stt.cost(3600, keyterms=False) == pytest.approx(0.20)
    assert stt.cost(1800, keyterms=True) == pytest.approx(0.125)
    tts = TextToSpeechPricing(per_1k_chars=Decimal("0.04"))
    assert tts.cost(250) == pytest.approx(0.01)


def test_unknown_models_are_unpriced_not_free():
    assert (
        estimate_cost("nope", prompt_tokens=10, cached_prompt_tokens=0, completion_tokens=10)
        is None
    )
    assert estimate_stt_cost("nope", audio_s=10, keyterms=False) is None
    assert estimate_tts_cost("nope", chars=10) is None
