"""Central registry of model prices (LLM and speech), used for cost *estimates* in traces.

Prices are a property of the model, not of a deployment, so they live here in code
(reviewed, versioned, tested) rather than in environment configuration. Adding a model
means adding one entry to the matching registry. A model without an entry is priced as
``None``, never as a misleading zero.

LLM: providers report cached prompt tokens as a *subset* of the prompt tokens, so the
uncached part is billed at the input rate and the cached part at the cached-input rate.

Speech (ElevenLabs API list prices, USD, as of November 2026): speech-to-text is billed
per hour of audio, plus a per-hour add-on when keyterm prompting is used; text-to-speech
per 1,000 characters of input text. Estimates are per second of audio: if the provider
rounds short clips up, real cost is higher, so treat STT figures as a lower bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

_PER_MTOK = Decimal(1_000_000)
_SECONDS_PER_HOUR = Decimal(3600)
_PER_KCHAR = Decimal(1000)


@dataclass(frozen=True)
class ModelPricing:
    """USD per 1M tokens."""

    input: Decimal
    cached_input: Decimal
    output: Decimal

    def cost(self, prompt_tokens: int, cached_prompt_tokens: int, completion_tokens: int) -> float:
        cached = max(0, min(cached_prompt_tokens, prompt_tokens))  # never trust it blindly
        uncached = prompt_tokens - cached
        total = uncached * self.input + cached * self.cached_input + completion_tokens * self.output
        return float(total / _PER_MTOK)


MODEL_PRICING: dict[str, ModelPricing] = {
    "mistral-small-latest": ModelPricing(
        input=Decimal("0.15"), cached_input=Decimal("0.015"), output=Decimal("0.60")
    ),
}


def pricing_for(model: str) -> ModelPricing | None:
    return MODEL_PRICING.get(model)


def estimate_cost(
    model: str, *, prompt_tokens: int, cached_prompt_tokens: int, completion_tokens: int
) -> float | None:
    """Estimated USD cost of one call, or ``None`` when the model has no registered price."""
    pricing = pricing_for(model)
    if pricing is None:
        return None
    return pricing.cost(prompt_tokens, cached_prompt_tokens, completion_tokens)


# ------------------------------------------------------------------ speech


@dataclass(frozen=True)
class SpeechToTextPricing:
    """USD per hour of audio."""

    per_hour: Decimal
    keyterms_per_hour: Decimal = Decimal(0)

    def cost(self, audio_s: float, *, keyterms: bool) -> float:
        rate = self.per_hour + (self.keyterms_per_hour if keyterms else Decimal(0))
        return float(Decimal(str(max(0.0, audio_s))) * rate / _SECONDS_PER_HOUR)


@dataclass(frozen=True)
class TextToSpeechPricing:
    """USD per 1,000 characters of input text."""

    per_1k_chars: Decimal

    def cost(self, chars: int) -> float:
        return float(max(0, chars) * self.per_1k_chars / _PER_KCHAR)


STT_PRICING: dict[str, SpeechToTextPricing] = {
    "scribe_v2": SpeechToTextPricing(per_hour=Decimal("0.22"), keyterms_per_hour=Decimal("0.05")),
}

TTS_PRICING: dict[str, TextToSpeechPricing] = {
    "eleven_flash_v2_5": TextToSpeechPricing(per_1k_chars=Decimal("0.04")),
}


def estimate_stt_cost(model: str, *, audio_s: float, keyterms: bool) -> float | None:
    """Estimated USD cost of transcribing ``audio_s`` seconds, or ``None`` if unpriced."""
    pricing = STT_PRICING.get(model)
    return None if pricing is None else pricing.cost(audio_s, keyterms=keyterms)


def estimate_tts_cost(model: str, *, chars: int) -> float | None:
    """Estimated USD cost of synthesizing ``chars`` characters, or ``None`` if unpriced."""
    pricing = TTS_PRICING.get(model)
    return None if pricing is None else pricing.cost(chars)
