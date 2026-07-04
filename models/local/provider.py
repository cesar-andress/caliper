"""Local open-weight model provider."""

from __future__ import annotations

import time
from typing import Any

import structlog

from caliper.models.api_common import build_dry_run_response, is_provider_dry_run
from caliper.models.base import BaseModelProvider
from caliper.models.cost import CostEstimator, CostPricing
from caliper.models.errors import ProviderGenerationError, ProviderUnavailableError
from caliper.models.local.backends import LocalBackend, create_local_backend
from caliper.models.local.config import LocalModelSettings
from caliper.models.local.metadata import (
    NvmlSampler,
    collect_gpu_metadata,
    snapshot_gpu_memory,
)
from caliper.models.registry import register_provider
from caliper.models.retry import ProviderRuntimeConfig
from caliper.models.types import ModelRequest, ModelResponse

logger = structlog.get_logger(__name__)


@register_provider("local")
class LocalModelProvider(BaseModelProvider):
    """Run open-weight models locally via transformers, llama.cpp, or vLLM."""

    provider_type = "local"

    def __init__(
        self,
        *,
        model_name: str,
        provider_name: str = "local",
        model_path: str | None = None,
        runtime: ProviderRuntimeConfig | None = None,
        dry_run: bool | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            provider_name=provider_name,
            runtime=runtime,
            **config,
        )
        merged = dict(config)
        if model_path is not None:
            merged["model_path"] = model_path
        self.settings = LocalModelSettings.from_config(config=merged)
        self.dry_run = dry_run if dry_run is not None else is_provider_dry_run(config=merged)
        self.cost_estimator = CostEstimator(CostPricing.from_config(merged))
        self._backend: LocalBackend | None = None
        self._gpu_metadata = collect_gpu_metadata(device=self.settings.device)
        self._nvml = NvmlSampler(
            device_index=self.settings.nvml_device_index,
            enabled=self.settings.nvml,
        )
        self._logged_startup = False

    def is_available(self) -> bool:
        if self.dry_run:
            return True
        if not self.settings.model_path:
            return False
        try:
            self.settings.validate_model_path()
        except (ValueError, FileNotFoundError):
            return False
        return True

    def _ensure_ready(self) -> None:
        if self.dry_run:
            return
        if not self.is_available():
            msg = (
                f"Provider '{self.provider_name}' (local) is not available: "
                "configure model_path in YAML or LOCAL_MODEL_PATH"
            )
            raise ProviderUnavailableError(msg, provider_name=self.provider_name)

    def _get_backend(self) -> LocalBackend:
        if self._backend is None:
            self._backend = create_local_backend(self.settings)
        return self._backend

    def _log_startup_metadata(self) -> None:
        if self._logged_startup:
            return
        logger.info(
            "local.provider.start",
            provider=self.provider_name,
            model_name=self.model_name,
            model_path=self.settings.model_path,
            nvml_enabled=self.settings.nvml and self._nvml.available,
            **self.settings.quantization_metadata,
            **self._gpu_metadata.to_dict(),
        )
        self._logged_startup = True

    def _generate_once(self, request: ModelRequest) -> ModelResponse:
        self._ensure_ready()
        self._log_startup_metadata()

        if self.dry_run:
            return build_dry_run_response(
                provider_type=self.provider_type,
                provider_name=self.provider_name,
                model_name=self.model_name,
                request=request,
                cost_estimator=self.cost_estimator,
            )

        backend = self._get_backend()
        started = time.perf_counter()

        def _run_inference() -> Any:
            backend.ensure_loaded()
            return backend.generate(request)

        if self.settings.nvml and self._nvml.available:
            result, nvml_reading = self._nvml.measure(_run_inference)
        else:
            result = _run_inference()
            nvml_reading = None

        wall_latency_ms = (time.perf_counter() - started) * 1000
        mem_snapshot = snapshot_gpu_memory(device_index=self._gpu_metadata.device_index)

        raw_metadata: dict[str, Any] = {
            "provider_type": self.provider_type,
            "dry_run": False,
            "model_path": self.settings.model_path,
            "logical_model_name": self.model_name,
            "backend": self.settings.backend,
            "deterministic": self.settings.deterministic,
            "inference_latency_ms": result.inference_latency_ms,
            "wall_latency_ms": wall_latency_ms,
            **self.settings.quantization_metadata,
            **self._gpu_metadata.to_dict(),
            **mem_snapshot,
        }
        if result.metadata:
            raw_metadata.update(result.metadata)
        if nvml_reading is not None:
            raw_metadata.update(nvml_reading.to_dict())

        return ModelResponse(
            text=result.text,
            model_name=self.model_name,
            provider_name=self.provider_name,
            prompt_id=request.prompt_id,
            task_id=request.task_id,
            run_id=request.run_id,
            temperature=request.temperature,
            seed=request.seed,
            latency_ms=result.inference_latency_ms or wall_latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            raw_metadata=raw_metadata,
        )

    def unload(self) -> None:
        if self._backend is not None:
            self._backend.unload()
            self._backend = None
        self._nvml.shutdown()
