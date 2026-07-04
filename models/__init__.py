"""Model abstractions, registry, and built-in providers."""

from caliper.models.base import BaseModelProvider
from caliper.models.errors import (
    ProviderError,
    ProviderGenerationError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from caliper.models.registry import (
    create_provider,
    get_provider,
    get_provider_class,
    list_provider_types,
    register_provider,
    register_provider_alias,
)
from caliper.models.retry import ProviderRuntimeConfig, RetryPolicy
from caliper.models.types import ModelRequest, ModelResponse

# Register built-in providers on import.
from caliper.models import anthropic_provider as _anthropic_provider  # noqa: F401
from caliper.models import gemini_provider as _gemini_provider  # noqa: F401
from caliper.models import mock as _mock  # noqa: F401
from caliper.models import openai_provider as _openai_provider  # noqa: F401
from caliper.models import random_provider as _random_provider  # noqa: F401
from caliper.models.local import provider as _local_provider  # noqa: F401

__all__ = [
    "AnthropicProvider",
    "BaseModelProvider",
    "GeminiProvider",
    "LocalModelProvider",
    "MockProvider",
    "ModelRequest",
    "ModelResponse",
    "OpenAIProvider",
    "ProviderError",
    "ProviderGenerationError",
    "ProviderRuntimeConfig",
    "ProviderTimeoutError",
    "ProviderUnavailableError",
    "RandomProvider",
    "RetryPolicy",
    "create_provider",
    "get_provider",
    "get_provider_class",
    "list_provider_types",
    "register_provider",
    "register_provider_alias",
]

from caliper.models.anthropic_provider import AnthropicProvider
from caliper.models.gemini_provider import GeminiProvider
from caliper.models.local.provider import LocalModelProvider
from caliper.models.mock import MockProvider
from caliper.models.openai_provider import OpenAIProvider
from caliper.models.random_provider import RandomProvider
