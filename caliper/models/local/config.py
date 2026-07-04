"""Configuration for local model providers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

LocalBackendName = Literal["transformers", "llama_cpp", "vllm"]
QuantizationMode = Literal["none", "4bit", "8bit", "awq", "gptq", "gguf"]


@dataclass(frozen=True)
class LocalModelSettings:
    """Resolved settings for a local model provider."""

    backend: LocalBackendName = "transformers"
    model_path: str = ""
    device: str = "cuda:0"
    dtype: str = "auto"
    quantization: QuantizationMode = "none"
    trust_remote_code: bool = False
    nvml: bool = False
    nvml_device_index: int = 0
    deterministic: bool = True
    # transformers / vLLM
    gpu_memory_utilization: float = 0.90
    # llama.cpp
    n_gpu_layers: int = -1
    n_ctx: int = 4096
    # vLLM
    tensor_parallel_size: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        *,
        config: dict[str, Any],
        model_path_override: str | None = None,
    ) -> LocalModelSettings:
        env_path = os.environ.get("LOCAL_MODEL_PATH", "").strip()
        resolved_path = (
            model_path_override
            or config.get("model_path")
            or env_path
        )
        backend = config.get("backend", "transformers")
        if backend not in {"transformers", "llama_cpp", "vllm"}:
            msg = f"unsupported local backend '{backend}'"
            raise ValueError(msg)

        quantization = config.get("quantization", "none")
        if backend == "llama_cpp" and quantization == "none":
            quantization = "gguf"

        known_keys = {
            "backend",
            "model_path",
            "device",
            "dtype",
            "quantization",
            "trust_remote_code",
            "nvml",
            "nvml_device_index",
            "deterministic",
            "gpu_memory_utilization",
            "n_gpu_layers",
            "n_ctx",
            "tensor_parallel_size",
            "dry_run",
            "timeout_seconds",
            "max_retries",
            "initial_backoff_seconds",
            "backoff_multiplier",
            "max_backoff_seconds",
        }
        extra = {k: v for k, v in config.items() if k not in known_keys}

        return cls(
            backend=backend,
            model_path=str(resolved_path),
            device=str(config.get("device", "cuda:0")),
            dtype=str(config.get("dtype", "auto")),
            quantization=quantization,
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            nvml=bool(config.get("nvml", False)),
            nvml_device_index=int(config.get("nvml_device_index", 0)),
            deterministic=bool(config.get("deterministic", True)),
            gpu_memory_utilization=float(config.get("gpu_memory_utilization", 0.90)),
            n_gpu_layers=int(config.get("n_gpu_layers", -1)),
            n_ctx=int(config.get("n_ctx", 4096)),
            tensor_parallel_size=int(config.get("tensor_parallel_size", 1)),
            extra=extra,
        )

    @property
    def quantization_metadata(self) -> dict[str, Any]:
        return {
            "quantization": self.quantization,
            "dtype": self.dtype,
            "backend": self.backend,
        }

    def validate_model_path(self) -> None:
        if not self.model_path:
            msg = (
                "local model_path is required; set providers.<name>.extra.model_path "
                "in YAML, model.extra.model_path, or LOCAL_MODEL_PATH"
            )
            raise ValueError(msg)
        path = Path(self.model_path)
        if path.exists() and path.is_dir():
            return
        if path.suffix == ".gguf" and path.exists():
            return
        if "/" in self.model_path or self.model_path.startswith("models/"):
            return
        if not path.exists():
            msg = f"local model_path does not exist: {self.model_path}"
            raise FileNotFoundError(msg)
