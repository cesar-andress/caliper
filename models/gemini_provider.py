"""Google Gemini API model provider."""

from __future__ import annotations

from typing import Any

from caliper.models.api_common import ApiProviderMixin, resolve_api_key, wrap_provider_error
from caliper.models.base import BaseModelProvider
from caliper.models.errors import ProviderGenerationError
from caliper.models.registry import register_provider, register_provider_alias
from caliper.models.retry import ProviderRuntimeConfig
from caliper.models.types import ModelRequest, ModelResponse


@register_provider("gemini")
class GeminiProvider(BaseModelProvider, ApiProviderMixin):
    """Provider for Google Gemini generate-content API."""

    provider_type = "gemini"
    default_api_key_env = "GEMINI_API_KEY"

    def __init__(
        self,
        *,
        model_name: str,
        provider_name: str = "gemini",
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
        if self.api_key is None and (api_key_env or self.default_api_key_env) == "GEMINI_API_KEY":
            self.api_key = resolve_api_key("GOOGLE_API_KEY")
        self._client: Any | None = None

    def is_available(self) -> bool:
        return ApiProviderMixin.is_available(self)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as exc:
            msg = "google-genai package is required; install with: pip install 'caliper[api]'"
            raise ProviderGenerationError(
                msg,
                provider_name=self.provider_name,
                retryable=False,
            ) from exc

        api_key = self.api_key
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            client_kwargs["http_options"] = {"base_url": self.base_url}
        self._client = genai.Client(**client_kwargs)
        return self._client

    def _generate_once(self, request: ModelRequest) -> ModelResponse:
        self._ensure_ready()
        if self.dry_run:
            return self._dry_run_response(request)

        try:
            from google import genai
            from google.genai import errors as genai_errors
            from google.genai import types
        except ImportError as exc:
            msg = "google-genai package is required; install with: pip install 'caliper[api]'"
            raise ProviderGenerationError(
                msg,
                provider_name=self.provider_name,
                retryable=False,
            ) from exc

        client = self._get_client()
        config_kwargs: dict[str, Any] = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_output_tokens": request.max_tokens,
        }
        if request.stop:
            config_kwargs["stop_sequences"] = request.stop
        if request.seed is not None:
            config_kwargs["seed"] = request.seed

        generate_config = types.GenerateContentConfig(**config_kwargs)

        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=request.prompt,
                config=generate_config,
            )
        except genai_errors.ClientError as exc:
            retryable = getattr(exc, "code", None) in {408, 429, 500, 502, 503, 504}
            raise wrap_provider_error(
                exc,
                provider_name=self.provider_name,
                provider_type=self.provider_type,
                retryable=retryable,
            ) from exc
        except genai_errors.ServerError as exc:
            raise wrap_provider_error(
                exc,
                provider_name=self.provider_name,
                provider_type=self.provider_type,
                retryable=True,
            ) from exc
        except genai_errors.APIError as exc:
            raise wrap_provider_error(
                exc,
                provider_name=self.provider_name,
                provider_type=self.provider_type,
            ) from exc

        text = response.text or ""
        usage = response.usage_metadata
        prompt_tokens = getattr(usage, "prompt_token_count", None) if usage else None
        completion_tokens = getattr(usage, "candidates_token_count", None) if usage else None
        total_tokens = getattr(usage, "total_token_count", None) if usage else None

        raw_metadata = self._attach_cost_metadata(
            {
                "provider_type": self.provider_type,
                "dry_run": False,
                "response_id": getattr(response, "response_id", None),
                "model": self.model_name,
                "finish_reason": _extract_finish_reason(response),
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


register_provider_alias("google", "gemini")


def _extract_finish_reason(response: Any) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    first = candidates[0]
    return getattr(first, "finish_reason", None)
