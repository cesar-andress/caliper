"""Tests for model provider abstraction."""

from __future__ import annotations

import pytest

from caliper.models import (
    MockProvider,
    ModelRequest,
    ModelResponse,
    ProviderGenerationError,
    ProviderTimeoutError,
    RandomProvider,
    RetryPolicy,
    create_provider,
    list_provider_types,
)
from caliper.models.retry import ProviderRuntimeConfig


def _make_request(**overrides: object) -> ModelRequest:
    defaults = {
        "prompt": "What is 2+2?",
        "prompt_id": "zero-shot",
        "task_id": "gsm8k",
        "run_id": "run-001",
        "temperature": 0.0,
        "seed": 42,
    }
    defaults.update(overrides)
    return ModelRequest(**defaults)  # type: ignore[arg-type]


class TestMockProvider:
    def test_deterministic_output(self) -> None:
        provider = MockProvider(model_name="mock-v1", simulated_latency_ms=0)
        req = _make_request()
        r1 = provider.generate(req)
        r2 = provider.generate(req)
        assert r1.text == r2.text

    def test_match_rate_returns_expected_output(self) -> None:
        provider = MockProvider(model_name="mock-v1", simulated_latency_ms=0, match_rate=1.0)
        req = _make_request(metadata={"expected_output": "def add(a, b):\n    return a + b"})
        response = provider.generate(req)
        assert response.text.startswith("def add")

    def test_different_seed_different_output(self) -> None:
        provider = MockProvider(model_name="mock-v1", simulated_latency_ms=0)
        r1 = provider.generate(_make_request(seed=1))
        r2 = provider.generate(_make_request(seed=2))
        assert r1.text != r2.text

    def test_response_has_required_fields(self) -> None:
        provider = MockProvider(model_name="mock-v1", provider_name="mock", simulated_latency_ms=0)
        response = provider.generate(_make_request())
        assert isinstance(response, ModelResponse)
        assert response.text.startswith("[mock:")
        assert response.model_name == "mock-v1"
        assert response.provider_name == "mock"
        assert response.prompt_id == "zero-shot"
        assert response.task_id == "gsm8k"
        assert response.run_id == "run-001"
        assert response.temperature == 0.0
        assert response.seed == 42
        assert response.latency_ms >= 0
        assert response.prompt_tokens is not None
        assert response.completion_tokens is not None
        assert response.total_tokens is not None
        assert response.raw_metadata["deterministic"] is True

    def test_to_log_dict(self) -> None:
        provider = MockProvider(model_name="mock-v1", simulated_latency_ms=0)
        response = provider.generate(_make_request())
        logged = response.to_log_dict()
        assert logged["run_id"] == "run-001"
        assert "text" in logged


class TestRandomProvider:
    def test_stochastic_output_differs_across_calls(self) -> None:
        provider = RandomProvider(model_name="rand-v1", simulated_latency_ms=0)
        req = _make_request()
        texts = {provider.generate(req).text for _ in range(5)}
        assert len(texts) > 1

    def test_different_run_ids_produce_distinct_loggable_responses(self) -> None:
        provider = RandomProvider(model_name="rand-v1", simulated_latency_ms=0)
        base = {"prompt": "Q?", "prompt_id": "p1", "task_id": "t1", "temperature": 0.7, "seed": 99}

        r1 = provider.generate(ModelRequest(**base, run_id="run-A"))  # type: ignore[arg-type]
        r2 = provider.generate(ModelRequest(**base, run_id="run-B"))  # type: ignore[arg-type]

        assert r1.text != r2.text
        assert r1.run_id == "run-A"
        assert r2.run_id == "run-B"
        assert r1.raw_metadata["deterministic"] is False
        assert r2.raw_metadata["deterministic"] is False
        assert r1.to_log_dict()["run_id"] != r2.to_log_dict()["run_id"]

    def test_response_has_required_fields(self) -> None:
        provider = RandomProvider(model_name="rand-v1", provider_name="random", simulated_latency_ms=0)
        response = provider.generate(_make_request())
        assert response.model_name == "rand-v1"
        assert response.provider_name == "random"
        assert response.prompt_tokens is not None
        assert response.total_tokens is not None
        assert "call_counter" in response.raw_metadata


class TestRegistry:
    def test_builtin_providers_registered(self) -> None:
        types = list_provider_types()
        assert "mock" in types
        assert "random" in types
        assert "openai" in types
        assert "anthropic" in types
        assert "gemini" in types
        assert "local" in types

    def test_create_provider_factory(self) -> None:
        provider = create_provider("mock", model_name="via-factory", simulated_latency_ms=0)
        assert isinstance(provider, MockProvider)
        response = provider.generate(_make_request())
        assert response.model_name == "via-factory"

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown provider type"):
            create_provider("nonexistent", model_name="x")


class TestRetryAndTimeout:
    def test_retry_recovers_from_transient_failure(self) -> None:
        runtime = ProviderRuntimeConfig(
            timeout_seconds=5.0,
            retry=RetryPolicy(max_retries=2, initial_backoff_seconds=0.0),
        )
        provider = MockProvider(
            model_name="flaky",
            simulated_latency_ms=0,
            fail_attempts=2,
            runtime=runtime,
        )
        response = provider.generate(_make_request())
        assert response.text.startswith("[mock:")

    def test_retry_exhausted_raises(self) -> None:
        runtime = ProviderRuntimeConfig(
            timeout_seconds=5.0,
            retry=RetryPolicy(max_retries=1, initial_backoff_seconds=0.0),
        )
        provider = MockProvider(
            model_name="flaky",
            simulated_latency_ms=0,
            fail_attempts=5,
            runtime=runtime,
        )
        with pytest.raises(ProviderGenerationError, match="Simulated transient failure"):
            provider.generate(_make_request())

    def test_timeout_raises(self) -> None:
        runtime = ProviderRuntimeConfig(timeout_seconds=0.001, retry=RetryPolicy(max_retries=0))
        provider = MockProvider(
            model_name="slow",
            simulated_latency_ms=50,
            runtime=runtime,
        )
        with pytest.raises(ProviderTimeoutError, match="exceeded timeout"):
            provider.generate(_make_request())

    def test_batch_generate(self) -> None:
        provider = MockProvider(model_name="mock-v1", simulated_latency_ms=0)
        requests = [_make_request(run_id=f"run-{i}") for i in range(3)]
        responses = provider.generate_batch(requests)
        assert len(responses) == 3
        run_ids = {r.run_id for r in responses}
        assert run_ids == {"run-0", "run-1", "run-2"}
