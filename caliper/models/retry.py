"""Timeout and retry policy for model providers."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable, TypeVar

from pydantic import BaseModel, Field

from caliper.models.errors import ProviderError, ProviderGenerationError, ProviderTimeoutError

if TYPE_CHECKING:
    from caliper.models.types import ModelRequest, ModelResponse

T = TypeVar("T")


class RetryPolicy(BaseModel):
    """Configuration for retrying failed generation attempts."""

    max_retries: int = Field(default=0, ge=0)
    initial_backoff_seconds: float = Field(default=0.1, ge=0.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
    max_backoff_seconds: float = Field(default=30.0, ge=0.0)


class ProviderRuntimeConfig(BaseModel):
    """Runtime settings applied to every generation call."""

    timeout_seconds: float = Field(default=60.0, gt=0.0)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)


def execute_with_retry_and_timeout(
    operation: Callable[[], T],
    *,
    provider_name: str,
    timeout_seconds: float,
    retry: RetryPolicy,
) -> T:
    """Execute ``operation`` with timeout enforcement and configurable retries.

    No network I/O is performed by this helper; it wraps synchronous callables
    and enforces elapsed-time limits plus exponential backoff between retries.
    """
    last_error: ProviderError | None = None
    backoff = retry.initial_backoff_seconds

    for attempt in range(retry.max_retries + 1):
        started = time.perf_counter()
        try:
            result = operation()
        except ProviderError as exc:
            last_error = exc
            if attempt >= retry.max_retries or not getattr(exc, "retryable", False):
                raise
            time.sleep(min(backoff, retry.max_backoff_seconds))
            backoff = min(backoff * retry.backoff_multiplier, retry.max_backoff_seconds)
            continue

        elapsed = time.perf_counter() - started
        if elapsed > timeout_seconds:
            msg = (
                f"Provider '{provider_name}' exceeded timeout of {timeout_seconds}s "
                f"(elapsed {elapsed:.3f}s)"
            )
            raise ProviderTimeoutError(
                msg,
                provider_name=provider_name,
                timeout_seconds=timeout_seconds,
            )
        return result

    assert last_error is not None
    raise last_error


def ensure_retryable(error: ProviderGenerationError) -> ProviderGenerationError:
    """Mark a generation error as retryable."""
    error.retryable = True
    return error
