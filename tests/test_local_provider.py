"""Unit tests for local model provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from caliper.models import LocalModelProvider, ModelRequest, ProviderUnavailableError, create_provider
from caliper.models.local.backends import LocalGenerationResult, LlamaCppBackend, TransformersBackend
from caliper.models.local.config import LocalModelSettings
from caliper.models.local.metadata import GpuMetadata, NvmlReading, collect_gpu_metadata


def _make_request(**overrides: object) -> ModelRequest:
    defaults = {
        "prompt": "def add(a, b):\n    ",
        "prompt_id": "zero-shot",
        "task_id": "code-gen",
        "run_id": "run-001",
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": 32,
    }
    defaults.update(overrides)
    return ModelRequest(**defaults)  # type: ignore[arg-type]


class TestLocalModelSettings:
    def test_from_config_defaults(self) -> None:
        settings = LocalModelSettings.from_config(
            config={"backend": "transformers", "model_path": "org/model"},
        )
        assert settings.backend == "transformers"
        assert settings.model_path == "org/model"
        assert settings.quantization == "none"
        assert settings.deterministic is True

    def test_llama_cpp_defaults_quantization_to_gguf(self) -> None:
        settings = LocalModelSettings.from_config(
            config={"backend": "llama_cpp", "model_path": "/models/model.gguf"},
        )
        assert settings.quantization == "gguf"

    def test_env_model_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LOCAL_MODEL_PATH", "/env/model")
        settings = LocalModelSettings.from_config(config={"backend": "transformers"})
        assert settings.model_path == "/env/model"


class TestLocalProviderRegistry:
    def test_local_registered(self) -> None:
        provider = create_provider(
            "local",
            model_name="local-test",
            model_path="org/test-model",
            dry_run=True,
        )
        assert isinstance(provider, LocalModelProvider)


class TestLocalProviderDryRun:
    def test_dry_run_without_model_path(self) -> None:
        provider = LocalModelProvider(model_name="local-test", dry_run=True)
        assert provider.is_available() is True
        response = provider.generate(_make_request())
        assert response.raw_metadata["dry_run"] is True
        assert response.text.startswith("[dry-run:local:")

    def test_unavailable_without_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LOCAL_MODEL_PATH", raising=False)
        provider = LocalModelProvider(model_name="local-test", dry_run=False)
        assert provider.is_available() is False
        with pytest.raises(ProviderUnavailableError, match="is not available"):
            provider.generate(_make_request())


class TestLocalProviderGeneration:
    @patch("caliper.models.local.provider.create_local_backend")
    def test_generate_attaches_metadata(self, mock_create_backend: MagicMock) -> None:
        mock_backend = MagicMock()
        mock_backend.generate.return_value = LocalGenerationResult(
            text="return x + y",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            inference_latency_ms=42.0,
            metadata={"generation_mode": "transformers.generate"},
        )
        mock_create_backend.return_value = mock_backend

        provider = LocalModelProvider(
            model_name="local-test",
            model_path="org/test-model",
            backend="transformers",
            quantization="4bit",
            dtype="bfloat16",
            nvml=False,
        )
        provider._gpu_metadata = GpuMetadata(
            gpu_available=True,
            device="cuda:0",
            device_index=0,
            gpu_name="NVIDIA GeForce RTX 4090",
            gpu_memory_total_gb=24.0,
        )

        response = provider.generate(_make_request())

        assert response.text == "return x + y"
        assert response.latency_ms == 42.0
        assert response.raw_metadata["backend"] == "transformers"
        assert response.raw_metadata["quantization"] == "4bit"
        assert response.raw_metadata["gpu_name"] == "NVIDIA GeForce RTX 4090"
        assert response.raw_metadata["inference_latency_ms"] == 42.0
        mock_backend.ensure_loaded.assert_called_once()

    @patch("caliper.models.local.provider.create_local_backend")
    @patch("caliper.models.local.provider.NvmlSampler")
    def test_nvml_metadata_when_enabled(
        self,
        mock_nvml_cls: MagicMock,
        mock_create_backend: MagicMock,
    ) -> None:
        mock_backend = MagicMock()
        mock_result = LocalGenerationResult(
            text="ok",
            inference_latency_ms=10.0,
        )

        mock_nvml = MagicMock()
        mock_nvml.available = True
        mock_nvml.measure.return_value = (
            mock_result,
            NvmlReading(
                available=True,
                power_draw_watts=220.0,
                energy_joules=2.2,
                duration_ms=10.0,
                samples=[210.0, 220.0],
            ),
        )
        mock_nvml_cls.return_value = mock_nvml
        mock_create_backend.return_value = mock_backend

        provider = LocalModelProvider(
            model_name="local-test",
            model_path="org/test-model",
            nvml=True,
        )
        response = provider.generate(_make_request())

        assert response.raw_metadata["nvml_available"] is True
        assert response.raw_metadata["energy_joules"] == 2.2
        mock_nvml.measure.assert_called_once()


class TestGpuMetadata:
    @patch("caliper.models.local.metadata.torch", create=True)
    def test_collect_gpu_metadata_when_cuda_available(self, mock_torch: MagicMock) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.version.cuda = "12.1"
        props = MagicMock()
        props.name = "NVIDIA GeForce RTX 4090"
        props.total_memory = 24 * 1024**3
        props.major = 8
        props.minor = 9
        mock_torch.cuda.get_device_properties.return_value = props

        meta = collect_gpu_metadata(device="cuda:0")
        assert meta.gpu_available is True
        assert meta.gpu_name == "NVIDIA GeForce RTX 4090"
        assert meta.gpu_compute_capability == "8.9"


class TestBuildProviderLocal:
    def test_build_local_provider_from_config(self, sample_config, tmp_path) -> None:
        from caliper.config.schema import ExperimentConfig, ModelConfig, ProviderConfig
        from caliper.runners.executor import build_provider

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        config = ExperimentConfig(
            **{
                **sample_config.model_dump(),
                "providers": {
                    "local-gpu": ProviderConfig(
                        type="local",
                        extra={
                            "backend": "transformers",
                            "model_path": str(model_dir),
                            "quantization": "none",
                        },
                    ),
                },
                "models": [
                    ModelConfig(
                        id="local-eval",
                        provider="local-gpu",
                        model_id="my-local-model",
                    )
                ],
            }
        )
        provider = build_provider(config, config.models[0])
        assert isinstance(provider, LocalModelProvider)
        assert provider.model_name == "my-local-model"
        assert provider.settings.model_path == str(model_dir)


class TestTransformersBackendUnit:
    def test_generate_greedy_when_temperature_zero(self, tmp_path) -> None:
        settings = LocalModelSettings.from_config(
            config={
                "backend": "transformers",
                "model_path": str(tmp_path),
                "deterministic": True,
            }
        )
        backend = TransformersBackend(settings)
        backend._loaded = True

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer.return_value = {"input_ids": MagicMock(shape=[1, 3])}
        mock_tokenizer.decode.return_value = "pass"
        mock_model.device = "cpu"
        mock_model.generate.return_value = MagicMock()
        mock_model.generate.return_value.__getitem__.return_value = MagicMock()
        mock_model.generate.return_value[0].__getitem__.return_value = [1, 2]

        backend._tokenizer = mock_tokenizer
        backend._model = mock_model

        with patch("caliper.models.local.backends.apply_deterministic_seed"):
            result = backend.generate(_make_request(temperature=0.0))

        assert result.text == "pass"
        gen_kwargs = mock_model.generate.call_args.kwargs
        assert gen_kwargs["do_sample"] is False
