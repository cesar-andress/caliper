"""Inference backends for local open-weight models."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from caliper.models.errors import ProviderGenerationError
from caliper.models.local.config import LocalModelSettings
from caliper.models.types import ModelRequest


@dataclass
class LocalGenerationResult:
    """Raw output from a local backend."""

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    inference_latency_ms: float = 0.0
    metadata: dict[str, Any] | None = None


class LocalBackend(ABC):
    """Abstract backend for on-device model inference."""

    backend_name: str

    def __init__(self, settings: LocalModelSettings) -> None:
        self.settings = settings
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory."""

    @abstractmethod
    def generate(self, request: ModelRequest) -> LocalGenerationResult:
        """Run a single generation request."""

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
            self._loaded = True

    def unload(self) -> None:
        self._loaded = False


def create_local_backend(settings: LocalModelSettings) -> LocalBackend:
    if settings.backend == "transformers":
        return TransformersBackend(settings)
    if settings.backend == "llama_cpp":
        return LlamaCppBackend(settings)
    if settings.backend == "vllm":
        return VllmBackend(settings)
    msg = f"unsupported local backend: {settings.backend}"
    raise ValueError(msg)


def apply_deterministic_seed(seed: int | None, *, enabled: bool) -> None:
    if not enabled or seed is None:
        return
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import transformers

        transformers.set_seed(seed)
    except ImportError:
        pass


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


class TransformersBackend(LocalBackend):
    backend_name = "transformers"

    def __init__(self, settings: LocalModelSettings) -> None:
        super().__init__(settings)
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            msg = "transformers backend requires torch and transformers; pip install 'caliper[local]'"
            raise ProviderGenerationError(msg, retryable=False) from exc

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": self.settings.trust_remote_code,
        }
        if self.settings.dtype != "auto":
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            if self.settings.dtype in dtype_map:
                load_kwargs["torch_dtype"] = dtype_map[self.settings.dtype]

        if self.settings.quantization in {"4bit", "8bit"}:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                msg = "4bit/8bit quantization requires bitsandbytes; pip install 'caliper[local]'"
                raise ProviderGenerationError(msg, retryable=False) from exc
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=self.settings.quantization == "4bit",
                load_in_8bit=self.settings.quantization == "8bit",
            )
            load_kwargs["device_map"] = "auto"
        elif self.settings.device != "cpu":
            load_kwargs["device_map"] = self.settings.device

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.settings.model_path,
            trust_remote_code=self.settings.trust_remote_code,
        )
        if self.settings.quantization == "awq":
            try:
                from awq import AutoAWQForCausalLM  # type: ignore[import-not-found]

                self._model = AutoAWQForCausalLM.from_quantized(
                    self.settings.model_path,
                    fuse_layers=True,
                    trust_remote_code=self.settings.trust_remote_code,
                )
            except ImportError as exc:
                msg = "AWQ models require autoawq; install per docs/local-models.md"
                raise ProviderGenerationError(msg, retryable=False) from exc
        elif self.settings.quantization == "gptq":
            try:
                from auto_gptq import AutoGPTQForCausalLM  # type: ignore[import-not-found]

                self._model = AutoGPTQForCausalLM.from_quantized(
                    self.settings.model_path,
                    device=self.settings.device,
                    trust_remote_code=self.settings.trust_remote_code,
                )
            except ImportError as exc:
                msg = "GPTQ models require auto-gptq; install per docs/local-models.md"
                raise ProviderGenerationError(msg, retryable=False) from exc
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.settings.model_path,
                **load_kwargs,
            )

    def generate(self, request: ModelRequest) -> LocalGenerationResult:
        self.ensure_loaded()
        apply_deterministic_seed(request.seed, enabled=self.settings.deterministic)

        started = time.perf_counter()
        assert self._tokenizer is not None and self._model is not None

        inputs = self._tokenizer(request.prompt, return_tensors="pt")
        device = getattr(self._model, "device", None)
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        prompt_len = int(inputs["input_ids"].shape[-1])
        gen_kwargs: dict[str, Any] = {"max_new_tokens": request.max_tokens}
        if request.temperature == 0.0 and self.settings.deterministic:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = max(request.temperature, 1e-5)
            gen_kwargs["top_p"] = request.top_p
        if request.seed is not None and self.settings.deterministic:
            gen_kwargs["seed"] = request.seed

        output_ids = self._model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[0][prompt_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        latency_ms = (time.perf_counter() - started) * 1000

        completion_tokens = len(new_tokens)
        return LocalGenerationResult(
            text=text,
            prompt_tokens=prompt_len,
            completion_tokens=completion_tokens,
            total_tokens=prompt_len + completion_tokens,
            inference_latency_ms=latency_ms,
            metadata={"generation_mode": "transformers.generate"},
        )


class LlamaCppBackend(LocalBackend):
    backend_name = "llama_cpp"

    def __init__(self, settings: LocalModelSettings) -> None:
        super().__init__(settings)
        self._llm: Any = None

    def load(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            msg = "llama_cpp backend requires llama-cpp-python; pip install 'caliper[local-llama-cpp]'"
            raise ProviderGenerationError(msg, retryable=False) from exc

        seed = 0 if self.settings.deterministic else -1
        self._llm = Llama(
            model_path=self.settings.model_path,
            n_gpu_layers=self.settings.n_gpu_layers,
            n_ctx=self.settings.n_ctx,
            seed=seed,
            verbose=False,
        )

    def generate(self, request: ModelRequest) -> LocalGenerationResult:
        self.ensure_loaded()
        started = time.perf_counter()
        assert self._llm is not None

        kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "max_tokens": request.max_tokens,
            "temperature": max(request.temperature, 0.0),
            "top_p": request.top_p,
            "echo": False,
        }
        if request.stop:
            kwargs["stop"] = request.stop
        if request.seed is not None and self.settings.deterministic:
            kwargs["seed"] = request.seed

        result = self._llm(**kwargs)
        latency_ms = (time.perf_counter() - started) * 1000
        choice = result["choices"][0]
        text = choice.get("text", "")
        usage = result.get("usage", {})

        return LocalGenerationResult(
            text=text,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            inference_latency_ms=latency_ms,
            metadata={
                "generation_mode": "llama_cpp",
                "finish_reason": choice.get("finish_reason"),
            },
        )


class VllmBackend(LocalBackend):
    backend_name = "vllm"

    def __init__(self, settings: LocalModelSettings) -> None:
        super().__init__(settings)
        self._llm: Any = None

    def load(self) -> None:
        try:
            from vllm import LLM
        except ImportError as exc:
            msg = "vllm backend requires vllm; pip install 'caliper[local-vllm]'"
            raise ProviderGenerationError(msg, retryable=False) from exc

        self._llm = LLM(
            model=self.settings.model_path,
            tensor_parallel_size=self.settings.tensor_parallel_size,
            gpu_memory_utilization=self.settings.gpu_memory_utilization,
            trust_remote_code=self.settings.trust_remote_code,
            dtype=self.settings.dtype if self.settings.dtype != "auto" else "auto",
            seed=0 if self.settings.deterministic else None,
        )

    def generate(self, request: ModelRequest) -> LocalGenerationResult:
        self.ensure_loaded()
        try:
            from vllm import SamplingParams
        except ImportError as exc:
            msg = "vllm backend requires vllm; pip install 'caliper[local-vllm]'"
            raise ProviderGenerationError(msg, retryable=False) from exc

        started = time.perf_counter()
        assert self._llm is not None

        params_kwargs: dict[str, Any] = {
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "stop": request.stop or None,
        }
        if request.temperature == 0.0 and self.settings.deterministic:
            params_kwargs["temperature"] = 0.0
        else:
            params_kwargs["temperature"] = request.temperature
        if request.seed is not None and self.settings.deterministic:
            params_kwargs["seed"] = request.seed

        params = SamplingParams(**params_kwargs)
        outputs = self._llm.generate([request.prompt], params)
        latency_ms = (time.perf_counter() - started) * 1000

        completion = outputs[0].outputs[0]
        text = completion.text
        prompt_tokens = len(outputs[0].prompt_token_ids)
        completion_tokens = len(completion.token_ids)

        return LocalGenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            inference_latency_ms=latency_ms,
            metadata={
                "generation_mode": "vllm",
                "finish_reason": completion.finish_reason,
            },
        )
