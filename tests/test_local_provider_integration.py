"""Optional live tests for local model inference."""

from __future__ import annotations

import os

import pytest

from caliper.models import LocalModelProvider, ModelRequest

pytestmark = pytest.mark.local

requires_local_model = pytest.mark.skipif(
    not os.environ.get("LOCAL_MODEL_PATH"),
    reason="LOCAL_MODEL_PATH must be set",
)


def _make_request(**overrides: object) -> ModelRequest:
    defaults = {
        "prompt": "Say hello in one word.",
        "prompt_id": "integration",
        "task_id": "ping",
        "run_id": "local-integration",
        "temperature": 0.0,
        "max_tokens": 8,
        "seed": 0,
    }
    defaults.update(overrides)
    return ModelRequest(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def local_backend() -> str:
    return os.environ.get("CALIPER_LOCAL_BACKEND", "transformers")


@requires_local_model
def test_local_live_generation(local_backend: str) -> None:
    model_path = os.environ["LOCAL_MODEL_PATH"]
    provider = LocalModelProvider(
        model_name="local-integration",
        model_path=model_path,
        backend=local_backend,
        nvml=os.environ.get("CALIPER_LOCAL_NVML", "0") == "1",
    )
    response = provider.generate(_make_request())
    assert response.text
    assert response.raw_metadata["backend"] == local_backend
    assert response.raw_metadata["inference_latency_ms"] > 0
    provider.unload()
