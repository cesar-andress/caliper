"""Stochastic provider for variance and fragility testing."""

from __future__ import annotations

import hashlib
import random
import string
import time
from typing import Any

from caliper.models.base import BaseModelProvider
from caliper.models.registry import register_provider
from caliper.models.retry import ProviderRuntimeConfig
from caliper.models.types import ModelRequest, ModelResponse


@register_provider("random")
class RandomProvider(BaseModelProvider):
    """Stochastic provider that produces different outputs across runs.

    Each call derives entropy from ``run_id``, wall-clock time, and an
    optional request seed so repeated stochastic runs can be logged and
    distinguished by their experiment context.
    """

    provider_type = "random"

    def __init__(
        self,
        *,
        model_name: str,
        provider_name: str = "random",
        response_word_count: int = 8,
        simulated_latency_ms: float = 1.0,
        match_rate: float = 0.0,
        runtime: ProviderRuntimeConfig | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            provider_name=provider_name,
            runtime=runtime,
            **config,
        )
        self.response_word_count = response_word_count
        self.simulated_latency_ms = simulated_latency_ms
        self.match_rate = min(max(match_rate, 0.0), 1.0)
        self._call_counter = 0

    def is_available(self) -> bool:
        return True

    def _generate_once(self, request: ModelRequest) -> ModelResponse:
        self._call_counter += 1

        if self.simulated_latency_ms > 0:
            time.sleep(self.simulated_latency_ms / 1000.0)

        rng = self._build_rng(request)
        expected = request.metadata.get("expected_output")
        if self.match_rate > 0.0 and expected and rng.random() < self.match_rate:
            text = str(expected)
            completion_tokens = max(1, len(text.split()))
        else:
            words = [
                "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 8)))
                for _ in range(self.response_word_count)
            ]
            text = " ".join(words)
            completion_tokens = len(words)

        prompt_tokens = max(1, len(request.prompt.split()))

        return ModelResponse(
            text=text,
            model_name=self.model_name,
            provider_name=self.provider_name,
            prompt_id=request.prompt_id,
            task_id=request.task_id,
            run_id=request.run_id,
            temperature=request.temperature,
            seed=request.seed,
            latency_ms=self.simulated_latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            raw_metadata={
                "provider_type": self.provider_type,
                "deterministic": False,
                "call_counter": self._call_counter,
                "rng_seed": rng.randint(0, 2**31 - 1),
            },
        )

    def _build_rng(self, request: ModelRequest) -> random.Random:
        """Build a RNG unique to this call's experiment context."""
        entropy = (
            f"{request.run_id}:{request.prompt_id}:{request.task_id}:"
            f"{request.seed}:{time.time_ns()}:{self._call_counter}"
        )
        seed = int(hashlib.sha256(entropy.encode()).hexdigest()[:16], 16) % (2**32)
        return random.Random(seed)
