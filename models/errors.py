"""Model provider error hierarchy."""

from __future__ import annotations


class ProviderError(Exception):
    """Base exception for model provider failures."""

    def __init__(self, message: str, *, provider_name: str | None = None) -> None:
        self.provider_name = provider_name
        super().__init__(message)


class ProviderUnavailableError(ProviderError):
    """Raised when a provider is not configured or reachable."""


class ProviderTimeoutError(ProviderError):
    """Raised when generation exceeds the configured timeout."""

    def __init__(
        self,
        message: str,
        *,
        provider_name: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        super().__init__(message, provider_name=provider_name)


class ProviderGenerationError(ProviderError):
    """Raised when the provider fails to produce a valid response."""

    def __init__(
        self,
        message: str,
        *,
        provider_name: str | None = None,
        retryable: bool = False,
    ) -> None:
        self.retryable = retryable
        super().__init__(message, provider_name=provider_name)
