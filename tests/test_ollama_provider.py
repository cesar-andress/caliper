"""Unit tests for the Ollama local model provider (HTTP mocked)."""

from __future__ import annotations

from unittest.mock import patch

from pathlib import Path

import pytest

from caliper.config.loader import load_config, validate_config
from caliper.models import ModelRequest, OllamaProvider, ProviderGenerationError, ProviderUnavailableError, create_provider, list_provider_types
from caliper.models.ollama_client import OllamaConnectionError, OllamaHttpError, list_models
from caliper.models.retry import ProviderRuntimeConfig
from caliper.runners.experiment import ExperimentRunner

SMOKE_CONFIG = Path("configs/paper1/ollama_smoke.yaml")
PILOT_CONFIG = Path("configs/paper1/ollama_pilot_variance.yaml")


def _make_request(**overrides: object) -> ModelRequest:
    defaults = {
        "prompt": "Write def add(a,b): return a+b",
        "prompt_id": "direct",
        "task_id": "task-pilot-001",
        "run_id": "run-001",
        "temperature": 0.0,
        "seed": 42,
        "top_p": 1.0,
        "max_tokens": 128,
        "stop": ["\n\n"],
    }
    defaults.update(overrides)
    return ModelRequest(**defaults)  # type: ignore[arg-type]


class TestOllamaRegistry:
    def test_ollama_provider_registered(self) -> None:
        assert "ollama" in list_provider_types()
        provider = create_provider("ollama", model_name="qwen2.5-coder:7b")
        assert isinstance(provider, OllamaProvider)


class TestOllamaTags:
    @patch("caliper.models.ollama_client._request_json")
    def test_list_models_success(self, mock_request) -> None:
        mock_request.return_value = {
            "models": [
                {"name": "qwen2.5-coder:7b", "model": "qwen2.5-coder:7b"},
                {"name": "llama3.1:8b", "model": "llama3.1:8b"},
            ]
        }
        models = list_models(base_url="http://localhost:11434")
        assert len(models) == 2
        mock_request.assert_called_once()

    @patch("caliper.models.ollama_client._request_json")
    def test_list_models_connection_error(self, mock_request) -> None:
        mock_request.side_effect = OllamaConnectionError("Connection refused", url="http://localhost:11434/api/tags")
        with pytest.raises(OllamaConnectionError):
            list_models()


class TestOllamaGenerate:
    @patch("caliper.models.ollama_provider.ollama_generate")
    def test_successful_generate(self, mock_generate) -> None:
        mock_generate.return_value = {
            "model": "qwen2.5-coder:7b",
            "response": "def add(a, b):\n    return a + b",
            "done": True,
            "prompt_eval_count": 20,
            "eval_count": 12,
        }
        provider = OllamaProvider(
            model_name="qwen2.5-coder:7b",
            base_url="http://localhost:11434",
            runtime=ProviderRuntimeConfig(timeout_seconds=30.0),
        )
        provider._availability_checked = True
        provider._available = True

        response = provider.generate(_make_request())
        assert "def add" in response.text
        assert response.model_name == "qwen2.5-coder:7b"
        assert response.prompt_tokens == 20
        assert response.completion_tokens == 12
        assert response.total_tokens == 32
        assert response.raw_metadata["provider_type"] == "ollama"
        assert "ollama" in response.raw_metadata

        mock_generate.assert_called_once()
        kwargs = mock_generate.call_args.kwargs
        assert kwargs["model"] == "qwen2.5-coder:7b"
        assert kwargs["temperature"] == 0.0
        assert kwargs["top_p"] == 1.0
        assert kwargs["max_tokens"] == 128
        assert kwargs["seed"] == 42
        assert kwargs["stop"] == ["\n\n"]
        assert kwargs["think"] == "auto"

    @patch("caliper.models.ollama_provider.ollama_generate")
    def test_budget_exhausted_from_thinking(self, mock_generate) -> None:
        mock_generate.return_value = {
            "model": "qwen3:32b",
            "response": "",
            "thinking": "long chain of thought " * 50,
            "done": True,
            "done_reason": "length",
            "prompt_eval_count": 100,
            "eval_count": 1024,
        }
        provider = OllamaProvider(
            model_name="qwen3:32b",
            base_url="http://localhost:11434",
            runtime=ProviderRuntimeConfig(timeout_seconds=30.0),
        )
        provider._availability_checked = True
        provider._available = True
        response = provider.generate(_make_request(think=False, max_tokens=1024))
        assert response.text == ""
        assert response.budget_exhausted is True
        assert response.done_reason == "length"
        assert response.thinking_length > 0
        assert response.thinking_sha256
        kwargs = mock_generate.call_args.kwargs
        assert kwargs["think"] is False

    @patch("caliper.models.ollama_provider.ollama_list_models")
    @patch("caliper.models.ollama_client.generate")
    def test_ollama_unavailable(self, mock_generate, mock_list) -> None:
        mock_list.side_effect = OllamaConnectionError("Connection refused", url="http://localhost:11434/api/tags")
        provider = OllamaProvider(model_name="qwen2.5-coder:7b", runtime=ProviderRuntimeConfig(timeout_seconds=5.0))
        assert provider.is_available() is False
        with pytest.raises(ProviderUnavailableError, match="Ollama is not running"):
            provider.generate(_make_request())
        mock_generate.assert_not_called()

    @patch("caliper.models.ollama_provider.ollama_generate")
    def test_missing_model_error(self, mock_generate) -> None:
        mock_generate.side_effect = OllamaHttpError(
            404,
            '{"error":"model \'missing:7b\' not found"}',
            url="http://localhost:11434/api/generate",
        )
        provider = OllamaProvider(model_name="missing:7b", runtime=ProviderRuntimeConfig(timeout_seconds=5.0))
        provider._availability_checked = True
        provider._available = True
        with pytest.raises(ProviderGenerationError, match="not available locally"):
            provider.generate(_make_request())

    @patch("caliper.models.ollama_provider.ollama_generate")
    def test_generation_client_error(self, mock_generate) -> None:
        from caliper.models.ollama_client import OllamaResponseError

        mock_generate.side_effect = OllamaResponseError("unexpected payload")
        provider = OllamaProvider(model_name="qwen2.5-coder:7b", runtime=ProviderRuntimeConfig(timeout_seconds=5.0))
        provider._availability_checked = True
        provider._available = True
        with pytest.raises(ProviderGenerationError, match="Ollama generation failed"):
            provider.generate(_make_request())


class TestOllamaConfigs:
    def test_smoke_config_validates(self) -> None:
        assert validate_config(SMOKE_CONFIG) == []

    def test_pilot_config_validates(self) -> None:
        assert validate_config(PILOT_CONFIG) == []

    def test_smoke_expected_cells(self) -> None:
        config = load_config(SMOKE_CONFIG)
        assert config.total_combinations() == 12

    def test_pilot_expected_cells(self) -> None:
        config = load_config(PILOT_CONFIG)
        assert config.total_combinations() == 6000

    def test_smoke_dry_run(self) -> None:
        config = load_config(SMOKE_CONFIG)
        manifest = ExperimentRunner(config, config_path=SMOKE_CONFIG, dry_run=True).run()
        assert manifest.total_cells == 12

    def test_provider_type_alias_in_config(self) -> None:
        config = load_config(SMOKE_CONFIG)
        provider = config.providers["ollama_local"]
        assert provider.type == "ollama"
        assert provider.base_url == "http://localhost:11434"
        assert provider.timeout_seconds == 300


class TestOllamaCli:
    @patch("caliper.models.ollama_provider.list_local_models")
    def test_ollama_list_cli(self, mock_list) -> None:
        from click.testing import CliRunner

        from caliper.cli import main

        mock_list.return_value = ["qwen2.5-coder:7b", "llama3.1:8b"]
        result = CliRunner().invoke(main, ["ollama", "list"])
        assert result.exit_code == 0
        assert "qwen2.5-coder:7b" in result.output

    @patch("caliper.models.ollama_provider.list_local_models")
    def test_ollama_list_cli_connection_error(self, mock_list) -> None:
        from click.testing import CliRunner

        from caliper.cli import main

        mock_list.side_effect = OllamaConnectionError("Connection refused", url="http://localhost:11434/api/tags")
        result = CliRunner().invoke(main, ["ollama", "list"])
        assert result.exit_code != 0
        assert "Could not reach Ollama" in result.output
