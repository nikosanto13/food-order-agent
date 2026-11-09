"""Central registry of LLM token prices, used for cost *estimates* in traces.

Prices are a property of the model, not of a deployment, so they live here in code
(reviewed, versioned, tested) rather than in environment configuration. Adding a model
means adding one entry to ``MODEL_PRICING``.

Providers report cached prompt tokens as a *subset* of the prompt tokens, so the
uncached part is billed at the input rate and the cached part at the cached-input rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

_PER_MTOK = Decimal(1_000_000)


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
