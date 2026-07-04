"""Unit tests for API model providers (mocked SDKs)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from caliper.models import (
    AnthropicProvider,
    GeminiProvider,
    ModelRequest,
    OpenAIProvider,
    ProviderGenerationError,
    ProviderUnavailableError,
    create_provider,
    list_provider_types,
)
from caliper.models.cost import CostEstimator, CostPricing
from caliper.models.retry import ProviderRuntimeConfig, RetryPolicy


def _make_request(**overrides: object) -> ModelRequest:
    defaults = {
        "prompt": "Say hello in one word.",
        "prompt_id": "zero-shot",
        "task_id": "demo",
        "run_id": "run-001",
        "temperature": 0.0,
        "seed": 42,
        "max_tokens": 16,
    }
    defaults.update(overrides)
    return ModelRequest(**defaults)  # type: ignore[arg-type]


class TestRegistryApiProviders:
    def test_api_providers_registered(self) -> None:
        types = list_provider_types()
        assert "openai" in types
        assert "anthropic" in types
        assert "gemini" in types
        assert "google" in types

    def test_create_openai_provider(self) -> None:
        provider = create_provider(
            "openai",
            model_name="gpt-test",
            api_key="test-key",
            dry_run=True,
        )
        assert isinstance(provider, OpenAIProvider)
        assert provider.model_name == "gpt-test"


class TestAvailabilityAndDryRun:
    def test_unavailable_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = OpenAIProvider(model_name="gpt-test", api_key=None, dry_run=False)
        assert provider.is_available() is False
        with pytest.raises(ProviderUnavailableError, match="is not available"):
            provider.generate(_make_request())

    def test_dry_run_available_without_key(self) -> None:
        provider = OpenAIProvider(model_name="gpt-test", api_key=None, dry_run=True)
        assert provider.is_available() is True
        response = provider.generate(_make_request())
        assert response.raw_metadata["dry_run"] is True
        assert response.text.startswith("[dry-run:openai:")

    def test_dry_run_deterministic(self) -> None:
        provider = AnthropicProvider(model_name="claude-test", dry_run=True)
        r1 = provider.generate(_make_request())
        r2 = provider.generate(_make_request())
        assert r1.text == r2.text


class TestCostEstimation:
    def test_estimate_with_yaml_pricing(self) -> None:
        estimator = CostEstimator(
            CostPricing(input_per_million=3.0, output_per_million=15.0),
        )
        meta = estimator.metadata(prompt_tokens=1_000_000, completion_tokens=500_000)
        assert meta["pricing_configured"] is True
        assert meta["estimated_usd"] == pytest.approx(10.5)

    def test_cost_metadata_attached_in_dry_run(self) -> None:
        provider = OpenAIProvider(
            model_name="gpt-test",
            dry_run=True,
            cost_per_million_input_tokens=1.0,
            cost_per_million_output_tokens=2.0,
        )
        response = provider.generate(_make_request())
        assert response.raw_metadata["pricing_configured"] is True
        assert response.raw_metadata["estimated_usd"] is not None


class TestOpenAIProvider:
    @patch("openai.OpenAI")
    def test_generate_maps_response(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            id="resp-1",
            model="gpt-test",
            system_fingerprint="fp",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="hello"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        )

        provider = OpenAIProvider(
            model_name="gpt-test",
            api_key="sk-test",
            runtime=ProviderRuntimeConfig(timeout_seconds=30.0),
        )
        response = provider.generate(_make_request())

        assert response.text == "hello"
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 2
        assert response.total_tokens == 12
        assert response.raw_metadata["response_id"] == "resp-1"
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-test"

    @patch("openai.OpenAI")
    def test_rate_limit_is_retryable(self, mock_openai_cls: MagicMock) -> None:
        import openai

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.request = MagicMock()
        mock_client.chat.completions.create.side_effect = openai.RateLimitError(
            "rate limited",
            response=mock_response,
            body=None,
        )

        provider = OpenAIProvider(
            model_name="gpt-test",
            api_key="sk-test",
            runtime=ProviderRuntimeConfig(
                timeout_seconds=30.0,
                retry=RetryPolicy(max_retries=0),
            ),
        )
        with pytest.raises(ProviderGenerationError, match="rate limited") as exc_info:
            provider.generate(_make_request())
        assert exc_info.value.retryable is True


class TestAnthropicProvider:
    @patch("anthropic.Anthropic")
    def test_generate_maps_response(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = SimpleNamespace(
            id="msg-1",
            model="claude-test",
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="hi there")],
            usage=SimpleNamespace(input_tokens=8, output_tokens=3),
        )

        provider = AnthropicProvider(model_name="claude-test", api_key="sk-ant-test")
        response = provider.generate(_make_request())

        assert response.text == "hi there"
        assert response.prompt_tokens == 8
        assert response.completion_tokens == 3
        assert response.raw_metadata["response_id"] == "msg-1"


class TestGeminiProvider:
    @patch("google.genai.Client")
    def test_generate_maps_response(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.return_value = SimpleNamespace(
            text="gemini says hi",
            response_id="gem-1",
            usage_metadata=SimpleNamespace(
                prompt_token_count=11,
                candidates_token_count=4,
                total_token_count=15,
            ),
            candidates=[SimpleNamespace(finish_reason="STOP")],
        )

        provider = GeminiProvider(model_name="gemini-test", api_key="gem-test")
        response = provider.generate(_make_request())

        assert response.text == "gemini says hi"
        assert response.prompt_tokens == 11
        assert response.completion_tokens == 4
        assert response.total_tokens == 15

    def test_google_api_key_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "legacy-key")
        provider = GeminiProvider(model_name="gemini-test", dry_run=False)
        assert provider.is_available() is True
        assert provider.api_key == "legacy-key"

    def test_google_alias_factory(self) -> None:
        provider = create_provider("google", model_name="gemini-test", dry_run=True)
        assert isinstance(provider, GeminiProvider)


class TestBuildProvider:
    def test_build_api_provider_from_config(self, sample_config) -> None:
        from caliper.config.schema import ExperimentConfig, ModelConfig, ProviderConfig
        from caliper.runners.executor import build_provider

        config = ExperimentConfig(
            **{
                **sample_config.model_dump(),
                "providers": {
                    "openai-main": ProviderConfig(
                        type="openai",
                        api_key_env="OPENAI_API_KEY",
                        extra={"cost_per_million_input_tokens": 1.0},
                    ),
                },
                "models": [
                    ModelConfig(
                        id="gpt",
                        provider="openai-main",
                        model_id="gpt-custom-from-yaml",
                    )
                ],
            }
        )
        model = config.models[0]
        provider = build_provider(config, model)
        assert isinstance(provider, OpenAIProvider)
        assert provider.model_name == "gpt-custom-from-yaml"
        assert provider.api_key_env == "OPENAI_API_KEY"
