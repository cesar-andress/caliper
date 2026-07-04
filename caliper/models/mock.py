"""Deterministic mock provider for testing."""

from __future__ import annotations

import hashlib
import time
from typing import Any

from caliper.models.base import BaseModelProvider
from caliper.models.errors import ProviderGenerationError
from caliper.models.registry import register_provider
from caliper.models.retry import ProviderRuntimeConfig
from caliper.models.types import ModelRequest, ModelResponse


@register_provider("mock")
class MockProvider(BaseModelProvider):
    """Deterministic provider that never calls external APIs.

    Given the same prompt, seed, and model, ``MockProvider`` always returns
    the same text.  Useful for reproducible unit tests and dry-run pipelines.
    """

    provider_type = "mock"

    def __init__(
        self,
        *,
        model_name: str,
        provider_name: str = "mock",
        simulated_latency_ms: float = 1.0,
        fail_attempts: int = 0,
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
        self.simulated_latency_ms = simulated_latency_ms
        self.fail_attempts = fail_attempts
        self.match_rate = min(max(match_rate, 0.0), 1.0)
        self._attempt_counter = 0

    def is_available(self) -> bool:
        return True

    def _generate_once(self, request: ModelRequest) -> ModelResponse:
        if self._attempt_counter < self.fail_attempts:
            self._attempt_counter += 1
            msg = f"Simulated transient failure (attempt {self._attempt_counter})"
            raise ProviderGenerationError(
                msg,
                provider_name=self.provider_name,
                retryable=True,
            )

        if self.simulated_latency_ms > 0:
            time.sleep(self.simulated_latency_ms / 1000.0)

        text = self._deterministic_text(request)
        prompt_tokens = _estimate_tokens(request.prompt)
        completion_tokens = _estimate_tokens(text)

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
                "deterministic": True,
                "simulated_latency_ms": self.simulated_latency_ms,
            },
        )

    def _deterministic_text(self, request: ModelRequest) -> str:
        expected = request.metadata.get("expected_output")
        if self.match_rate > 0.0 and expected:
            if self._match_probability(request) < self.match_rate:
                return str(expected)
            return self._incorrect_output(request, str(expected))

        key = (
            f"{self.model_name}|{request.seed}|{request.temperature}|"
            f"{request.prompt_id}|{request.prompt}"
        )
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return f"[mock:{self.model_name}:{digest}]"

    def _match_probability(self, request: ModelRequest) -> float:
        key = (
            f"{self.model_name}|{request.seed}|{request.temperature}|"
            f"{request.prompt_id}|{request.task_id}|{request.run_id}"
        )
        digest = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        return (digest % 10_000) / 10_000.0

    def _incorrect_output(self, request: ModelRequest, expected: str) -> str:
        digest = hashlib.sha256(
            f"{request.task_id}|{request.prompt_id}|{request.seed}".encode()
        ).hexdigest()[:8]
        return f"{expected}\n# mock_incorrect:{digest}"


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))
