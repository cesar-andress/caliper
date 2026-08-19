"""Local Ollama model provider."""

from __future__ import annotations

import hashlib
import time
from typing import Any

from caliper.models.base import BaseModelProvider
from caliper.models.errors import ProviderGenerationError, ProviderUnavailableError
from caliper.models.ollama_client import (
    DEFAULT_OLLAMA_BASE_URL,
    OllamaClientError,
    OllamaConnectionError,
    OllamaHttpError,
    generate as ollama_generate,
    list_models as ollama_list_models,
)
from caliper.models.registry import register_provider
from caliper.models.retry import ProviderRuntimeConfig
from caliper.models.types import ModelRequest, ModelResponse


def _thinking_digest(thinking: str) -> tuple[int, str]:
    encoded = thinking.encode("utf-8", errors="replace")
    return len(thinking), hashlib.sha256(encoded).hexdigest()


def _is_budget_exhausted(*, text: str, done_reason: str | None, thinking: str) -> bool:
    if text.strip():
        return False
    if done_reason == "length":
        return True
    # Visible empty with non-empty thinking is the forensic failure mode.
    return bool(thinking.strip())


@register_provider("ollama")
class OllamaProvider(BaseModelProvider):
    """Provider for local models served by Ollama over HTTP."""

    provider_type = "ollama"

    def __init__(
        self,
        *,
        model_name: str,
        provider_name: str = "ollama",
        base_url: str | None = None,
        runtime: ProviderRuntimeConfig | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            provider_name=provider_name,
            runtime=runtime,
            **config,
        )
        self.base_url = base_url or config.get("base_url") or DEFAULT_OLLAMA_BASE_URL
        self._availability_checked = False
        self._available = False

    def is_available(self) -> bool:
        if self._availability_checked:
            return self._available
        try:
            ollama_list_models(base_url=self.base_url, timeout_seconds=min(self.runtime.timeout_seconds, 10.0))
        except OllamaConnectionError as exc:
            self._availability_checked = True
            self._available = False
            self._availability_error = (
                f"Ollama is not running at {self.base_url}. "
                f"Start Ollama and ensure the endpoint is reachable. ({exc})"
            )
            return False
        except OllamaClientError:
            self._availability_checked = True
            self._available = True
            return True
        self._availability_checked = True
        self._available = True
        return True

    def generate(self, request: ModelRequest) -> ModelResponse:
        if not self.is_available():
            message = getattr(
                self,
                "_availability_error",
                f"Ollama is not available at {self.base_url}",
            )
            raise ProviderUnavailableError(message, provider_name=self.provider_name)
        return super().generate(request)

    def _generate_once(self, request: ModelRequest) -> ModelResponse:
        if not self.is_available():
            message = getattr(self, "_availability_error", f"Ollama is not available at {self.base_url}")
            raise ProviderUnavailableError(message, provider_name=self.provider_name)

        started = time.perf_counter()
        try:
            payload = ollama_generate(
                base_url=self.base_url,
                model=self.model_name,
                prompt=request.prompt,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                seed=request.seed,
                stop=request.stop,
                timeout_seconds=self.runtime.timeout_seconds,
                think=request.think,
            )
        except OllamaConnectionError as exc:
            msg = (
                f"Ollama is not running at {self.base_url}. "
                f"Start Ollama and pull model '{self.model_name}' before running experiments."
            )
            raise ProviderUnavailableError(msg, provider_name=self.provider_name) from exc
        except OllamaHttpError as exc:
            message = exc.body.lower()
            if exc.status_code == 404 or "not found" in message:
                msg = (
                    f"Ollama model '{self.model_name}' is not available locally. "
                    f"Run: ollama pull {self.model_name}"
                )
                raise ProviderGenerationError(
                    msg,
                    provider_name=self.provider_name,
                    retryable=False,
                ) from exc
            raise ProviderGenerationError(
                f"Ollama request failed: {exc.body}",
                provider_name=self.provider_name,
                retryable=exc.status_code in {408, 429, 500, 502, 503, 504},
            ) from exc
        except OllamaClientError as exc:
            raise ProviderGenerationError(
                f"Ollama generation failed: {exc}",
                provider_name=self.provider_name,
                retryable=False,
            ) from exc

        text = str(payload.get("response", "") or "")
        thinking = str(payload.get("thinking", "") or "")
        done_reason = payload.get("done_reason")
        if done_reason is not None:
            done_reason = str(done_reason)
        prompt_tokens = payload.get("prompt_eval_count")
        completion_tokens = payload.get("eval_count")
        total_tokens = None
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            total_tokens = prompt_tokens + completion_tokens

        thinking_length, thinking_sha = _thinking_digest(thinking) if thinking else (0, None)
        budget_exhausted = _is_budget_exhausted(
            text=text, done_reason=done_reason, thinking=thinking
        )

        latency_ms = (time.perf_counter() - started) * 1000
        return ModelResponse(
            text=text,
            model_name=str(payload.get("model", self.model_name)),
            provider_name=self.provider_name,
            prompt_id=request.prompt_id,
            task_id=request.task_id,
            run_id=request.run_id,
            temperature=request.temperature,
            seed=request.seed,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
            total_tokens=total_tokens,
            done_reason=done_reason,
            thinking=thinking if thinking else None,
            thinking_length=thinking_length,
            thinking_sha256=thinking_sha,
            budget_exhausted=budget_exhausted,
            raw_metadata={
                "provider_type": self.provider_type,
                "base_url": self.base_url,
                "think_request": request.think,
                "num_predict": request.max_tokens,
                "ollama": payload,
            },
        )


def list_local_models(*, base_url: str = DEFAULT_OLLAMA_BASE_URL, timeout_seconds: float = 10.0) -> list[str]:
    """Return installed Ollama model names."""
    models = ollama_list_models(base_url=base_url, timeout_seconds=timeout_seconds)
    names: list[str] = []
    for model in models:
        name = model.get("name") or model.get("model")
        if isinstance(name, str) and name:
            names.append(name)
    return names
