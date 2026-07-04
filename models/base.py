"""Abstract base class for model providers."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from caliper.models.errors import ProviderUnavailableError
from caliper.models.retry import ProviderRuntimeConfig, execute_with_retry_and_timeout
from caliper.models.types import ModelRequest, ModelResponse


class BaseModelProvider(ABC):
    """Abstract interface for API and local model providers.

    Concrete implementations (OpenAI, Anthropic, Gemini, local vLLM, etc.)
    subclass this and register via ``register_provider``.  The public
    ``generate`` method applies timeout and retry policies; subclasses
    implement ``_generate_once`` only.
    """

    provider_type: str = "base"

    def __init__(
        self,
        *,
        model_name: str,
        provider_name: str,
        runtime: ProviderRuntimeConfig | None = None,
        **config: Any,
    ) -> None:
        self.model_name = model_name
        self.provider_name = provider_name
        self.runtime = runtime or ProviderRuntimeConfig()
        self.config = config

    @abstractmethod
    def _generate_once(self, request: ModelRequest) -> ModelResponse:
        """Perform a single generation attempt (no retry/timeout wrapper)."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the provider is configured and ready."""

    def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate a completion with timeout and retry enforcement."""
        if not self.is_available():
            msg = f"Provider '{self.provider_name}' ({self.provider_type}) is not available"
            raise ProviderUnavailableError(msg, provider_name=self.provider_name)

        started = time.perf_counter()

        def _attempt() -> ModelResponse:
            response = self._generate_once(request)
            elapsed_ms = (time.perf_counter() - started) * 1000
            if response.latency_ms == 0.0:
                return response.model_copy(update={"latency_ms": elapsed_ms})
            return response

        return execute_with_retry_and_timeout(
            _attempt,
            provider_name=self.provider_name,
            timeout_seconds=self.runtime.timeout_seconds,
            retry=self.runtime.retry,
        )

    def generate_batch(self, requests: list[ModelRequest]) -> list[ModelResponse]:
        """Generate completions for a batch of requests sequentially."""
        return [self.generate(req) for req in requests]
