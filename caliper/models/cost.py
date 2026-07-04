"""Cost estimation hooks for API model providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CostPricing:
    """Per-million-token pricing supplied via YAML provider config (not hardcoded)."""

    input_per_million: float | None = None
    output_per_million: float | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CostPricing:
        return cls(
            input_per_million=_coerce_float(
                config.get("cost_per_million_input_tokens")
                or config.get("input_cost_per_million")
            ),
            output_per_million=_coerce_float(
                config.get("cost_per_million_output_tokens")
                or config.get("output_cost_per_million")
            ),
        )

    @property
    def is_configured(self) -> bool:
        return self.input_per_million is not None or self.output_per_million is not None


@dataclass(frozen=True)
class CostEstimate:
    """Estimated cost for a single generation call."""

    prompt_tokens: int | None
    completion_tokens: int | None
    estimated_usd: float | None
    pricing_configured: bool

    def to_metadata(self) -> dict[str, Any]:
        return {
            "estimated_usd": self.estimated_usd,
            "pricing_configured": self.pricing_configured,
            "cost_input_per_million": None,
            "cost_output_per_million": None,
        }


class CostEstimator:
    """Estimate request cost from token usage and YAML-supplied pricing."""

    def __init__(self, pricing: CostPricing) -> None:
        self.pricing = pricing

    def estimate(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> CostEstimate:
        if prompt_tokens is None and completion_tokens is None:
            return CostEstimate(None, None, None, self.pricing.is_configured)

        if not self.pricing.is_configured:
            return CostEstimate(prompt_tokens, completion_tokens, None, False)

        prompt = prompt_tokens or 0
        completion = completion_tokens or 0
        input_rate = self.pricing.input_per_million or 0.0
        output_rate = self.pricing.output_per_million or 0.0
        estimated = (prompt * input_rate + completion * output_rate) / 1_000_000
        return CostEstimate(prompt_tokens, completion_tokens, estimated, True)

    def metadata(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> dict[str, Any]:
        estimate = self.estimate(prompt_tokens, completion_tokens)
        return {
            "estimated_usd": estimate.estimated_usd,
            "pricing_configured": estimate.pricing_configured,
            "cost_input_per_million": self.pricing.input_per_million,
            "cost_output_per_million": self.pricing.output_per_million,
        }


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
