"""Shared utilities for HTTP API model providers."""

from __future__ import annotations

import hashlib
import os
from typing import Any

from caliper.models.cost import CostEstimator, CostPricing
from caliper.models.errors import ProviderGenerationError, ProviderUnavailableError
from caliper.models.types import ModelRequest, ModelResponse


def resolve_api_key(api_key_env: str) -> str | None:
    """Read an API key from the named environment variable."""
    value = os.environ.get(api_key_env)
    if value is None or not value.strip():
        return None
    return value.strip()


def is_provider_dry_run(*, config: dict[str, Any]) -> bool:
    """Return True when provider calls should be skipped."""
    if config.get("dry_run"):
        return True
    env_flag = os.environ.get("CALIPER_PROVIDER_DRY_RUN", "").strip().lower()
    return env_flag in {"1", "true", "yes", "on"}


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def build_dry_run_response(
    *,
    provider_type: str,
    provider_name: str,
    model_name: str,
    request: ModelRequest,
    cost_estimator: CostEstimator,
) -> ModelResponse:
    """Return a deterministic placeholder response without calling an API."""
    key = (
        f"{provider_type}|{model_name}|{request.seed}|{request.temperature}|"
        f"{request.prompt_id}|{request.prompt}"
    )
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    text = f"[dry-run:{provider_type}:{model_name}:{digest}]"
    prompt_tokens = estimate_tokens(request.prompt)
    completion_tokens = estimate_tokens(text)
    cost_meta = cost_estimator.metadata(prompt_tokens, completion_tokens)

    return ModelResponse(
        text=text,
        model_name=model_name,
        provider_name=provider_name,
        prompt_id=request.prompt_id,
        task_id=request.task_id,
        run_id=request.run_id,
        temperature=request.temperature,
        seed=request.seed,
        latency_ms=0.0,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        raw_metadata={
            "provider_type": provider_type,
            "dry_run": True,
            "deterministic": True,
            **cost_meta,
        },
    )


def require_api_key(
    *,
    provider_name: str,
    provider_type: str,
    api_key_env: str,
    api_key: str | None,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    if api_key:
        return
    msg = (
        f"Provider '{provider_name}' ({provider_type}) is not available: "
        f"set the {api_key_env} environment variable"
    )
    raise ProviderUnavailableError(msg, provider_name=provider_name)


def map_http_status_to_retryable(status_code: int | None) -> bool:
    if status_code is None:
        return False
    return status_code in {408, 409, 429, 500, 502, 503, 504}


def wrap_provider_error(
    exc: Exception,
    *,
    provider_name: str,
    provider_type: str,
    retryable: bool | None = None,
) -> ProviderGenerationError:
    status_code = getattr(exc, "status_code", None)
    if retryable is None:
        retryable = map_http_status_to_retryable(status_code)
    message = f"{provider_type} generation failed: {exc}"
    return ProviderGenerationError(
        message,
        provider_name=provider_name,
        retryable=retryable,
    )


class ApiProviderMixin:
    """Mixin with shared initialization for API-backed providers."""

    provider_type: str
    default_api_key_env: str

    model_name: str
    provider_name: str
    api_key_env: str
    api_key: str | None
    base_url: str | None
    dry_run: bool
    cost_estimator: CostEstimator
    config: dict[str, Any]

    def _init_api_provider(
        self,
        *,
        model_name: str,
        provider_name: str,
        api_key_env: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        dry_run: bool | None = None,
        config: dict[str, Any],
    ) -> None:
        self.model_name = model_name
        self.provider_name = provider_name
        self.api_key_env = api_key_env or self.default_api_key_env
        self.base_url = base_url
        self.config = config
        self.dry_run = dry_run if dry_run is not None else is_provider_dry_run(config=config)
        resolved_key = api_key or resolve_api_key(self.api_key_env)
        self.api_key = resolved_key
        self.cost_estimator = CostEstimator(CostPricing.from_config(config))

    def is_available(self) -> bool:
        if self.dry_run:
            return True
        return self.api_key is not None

    def _ensure_ready(self) -> None:
        require_api_key(
            provider_name=self.provider_name,
            provider_type=self.provider_type,
            api_key_env=self.api_key_env,
            api_key=self.api_key,
            dry_run=self.dry_run,
        )

    def _dry_run_response(self, request: ModelRequest) -> ModelResponse:
        return build_dry_run_response(
            provider_type=self.provider_type,
            provider_name=self.provider_name,
            model_name=self.model_name,
            request=request,
            cost_estimator=self.cost_estimator,
        )

    def _attach_cost_metadata(
        self,
        raw_metadata: dict[str, Any],
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> dict[str, Any]:
        return {
            **raw_metadata,
            **self.cost_estimator.metadata(prompt_tokens, completion_tokens),
        }
