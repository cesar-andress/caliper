"""Anthropic API model provider."""

from __future__ import annotations

from typing import Any

from caliper.models.api_common import ApiProviderMixin, wrap_provider_error
from caliper.models.base import BaseModelProvider
from caliper.models.errors import ProviderGenerationError
from caliper.models.registry import register_provider
from caliper.models.retry import ProviderRuntimeConfig
from caliper.models.types import ModelRequest, ModelResponse


@register_provider("anthropic")
class AnthropicProvider(BaseModelProvider, ApiProviderMixin):
    """Provider for Anthropic Messages API."""

    provider_type = "anthropic"
    default_api_key_env = "ANTHROPIC_API_KEY"

    def __init__(
        self,
        *,
        model_name: str,
        provider_name: str = "anthropic",
        api_key_env: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        dry_run: bool | None = None,
        runtime: ProviderRuntimeConfig | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            provider_name=provider_name,
            runtime=runtime,
            **config,
        )
        self._init_api_provider(
            model_name=model_name,
            provider_name=provider_name,
            api_key_env=api_key_env,
            api_key=api_key,
            base_url=base_url,
            dry_run=dry_run,
            config=config,
        )
        self._client: Any | None = None

    def is_available(self) -> bool:
        return ApiProviderMixin.is_available(self)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:
            msg = "anthropic package is required; install with: pip install 'caliper[api]'"
            raise ProviderGenerationError(
                msg,
                provider_name=self.provider_name,
                retryable=False,
            ) from exc

        timeout = self.runtime.timeout_seconds
        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "max_retries": 0,
            "timeout": timeout,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self._client = anthropic.Anthropic(**client_kwargs)
        return self._client

    def _generate_once(self, request: ModelRequest) -> ModelResponse:
        self._ensure_ready()
        if self.dry_run:
            return self._dry_run_response(request)

        try:
            import anthropic
        except ImportError as exc:
            msg = "anthropic package is required; install with: pip install 'caliper[api]'"
            raise ProviderGenerationError(
                msg,
                provider_name=self.provider_name,
                retryable=False,
            ) from exc

        client = self._get_client()
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if request.stop:
            params["stop_sequences"] = request.stop

        try:
            message = client.messages.create(**params)
        except anthropic.RateLimitError as exc:
            raise wrap_provider_error(
                exc,
                provider_name=self.provider_name,
                provider_type=self.provider_type,
                retryable=True,
            ) from exc
        except (
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
            anthropic.OverloadedError,
        ) as exc:
            raise wrap_provider_error(
                exc,
                provider_name=self.provider_name,
                provider_type=self.provider_type,
                retryable=True,
            ) from exc
        except (
            anthropic.AuthenticationError,
            anthropic.BadRequestError,
            anthropic.PermissionDeniedError,
        ) as exc:
            raise wrap_provider_error(
                exc,
                provider_name=self.provider_name,
                provider_type=self.provider_type,
                retryable=False,
            ) from exc
        except anthropic.APIError as exc:
            raise wrap_provider_error(
                exc,
                provider_name=self.provider_name,
                provider_type=self.provider_type,
            ) from exc

        text_blocks = [
            block.text for block in message.content if getattr(block, "type", None) == "text"
        ]
        text = "".join(text_blocks)
        usage = message.usage
        prompt_tokens = usage.input_tokens if usage else None
        completion_tokens = usage.output_tokens if usage else None
        total_tokens = (
            (prompt_tokens or 0) + (completion_tokens or 0)
            if prompt_tokens is not None or completion_tokens is not None
            else None
        )

        raw_metadata = self._attach_cost_metadata(
            {
                "provider_type": self.provider_type,
                "dry_run": False,
                "response_id": message.id,
                "stop_reason": message.stop_reason,
                "model": message.model,
            },
            prompt_tokens,
            completion_tokens,
        )

        return ModelResponse(
            text=text,
            model_name=self.model_name,
            provider_name=self.provider_name,
            prompt_id=request.prompt_id,
            task_id=request.task_id,
            run_id=request.run_id,
            temperature=request.temperature,
            seed=request.seed,
            latency_ms=0.0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw_metadata=raw_metadata,
        )
