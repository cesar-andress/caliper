"""Optional live integration tests for API model providers.

These tests call real provider APIs and are skipped unless the corresponding
API key environment variable is set.
"""

from __future__ import annotations

import os

import pytest

from caliper.models import AnthropicProvider, GeminiProvider, ModelRequest, OpenAIProvider

pytestmark = pytest.mark.integration

requires_openai = pytest.mark.skipif(
    not (os.environ.get("OPENAI_API_KEY") and os.environ.get("CALIPER_OPENAI_MODEL")),
    reason="OPENAI_API_KEY and CALIPER_OPENAI_MODEL must be set",
)
requires_anthropic = pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("CALIPER_ANTHROPIC_MODEL")),
    reason="ANTHROPIC_API_KEY and CALIPER_ANTHROPIC_MODEL must be set",
)
requires_gemini = pytest.mark.skipif(
    not (
        (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        and os.environ.get("CALIPER_GEMINI_MODEL")
    ),
    reason="GEMINI_API_KEY (or GOOGLE_API_KEY) and CALIPER_GEMINI_MODEL must be set",
)


def _make_request(**overrides: object) -> ModelRequest:
    defaults = {
        "prompt": "Reply with exactly the word: pong",
        "prompt_id": "integration",
        "task_id": "ping",
        "run_id": "integration-run",
        "temperature": 0.0,
        "max_tokens": 8,
    }
    defaults.update(overrides)
    return ModelRequest(**defaults)  # type: ignore[arg-type]


@requires_openai
def test_openai_live_generation(openai_model: str) -> None:
    provider = OpenAIProvider(model_name=openai_model)
    response = provider.generate(_make_request())
    assert response.text
    assert response.model_name == openai_model
    assert response.raw_metadata["dry_run"] is False


@requires_anthropic
def test_anthropic_live_generation(anthropic_model: str) -> None:
    provider = AnthropicProvider(model_name=anthropic_model)
    response = provider.generate(_make_request())
    assert response.text
    assert response.model_name == anthropic_model


@requires_gemini
def test_gemini_live_generation(gemini_model: str) -> None:
    provider = GeminiProvider(model_name=gemini_model)
    response = provider.generate(_make_request())
    assert response.text
    assert response.model_name == gemini_model


@pytest.fixture
def openai_model() -> str:
    model = os.environ["CALIPER_OPENAI_MODEL"]
    return model


@pytest.fixture
def anthropic_model() -> str:
    model = os.environ["CALIPER_ANTHROPIC_MODEL"]
    return model


@pytest.fixture
def gemini_model() -> str:
    model = os.environ["CALIPER_GEMINI_MODEL"]
    return model
