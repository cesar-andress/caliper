"""Model provider registry and factory."""

from __future__ import annotations

from typing import Any

from caliper.models.base import BaseModelProvider
from caliper.models.errors import ProviderError
from caliper.models.retry import ProviderRuntimeConfig

_PROVIDERS: dict[str, type[BaseModelProvider]] = {}


def register_provider(provider_type: str):
    """Decorator to register a provider class under a type name."""

    def decorator(cls: type[BaseModelProvider]) -> type[BaseModelProvider]:
        if provider_type in _PROVIDERS:
            msg = f"Provider type '{provider_type}' is already registered"
            raise ProviderError(msg)
        cls.provider_type = provider_type
        _PROVIDERS[provider_type] = cls
        return cls

    return decorator


def get_provider_class(provider_type: str) -> type[BaseModelProvider]:
    """Retrieve a registered provider class by type name.

    Raises:
        KeyError: If no provider is registered under ``provider_type``.
    """
    if provider_type not in _PROVIDERS:
        registered = ", ".join(sorted(_PROVIDERS)) or "(none)"
        msg = f"Unknown provider type '{provider_type}'. Registered: {registered}"
        raise KeyError(msg)
    return _PROVIDERS[provider_type]


def register_provider_alias(alias: str, provider_type: str) -> None:
    """Register an alternate name for an existing provider type."""
    if provider_type not in _PROVIDERS:
        msg = f"Cannot alias unknown provider type '{provider_type}'"
        raise KeyError(msg)
    if alias in _PROVIDERS and alias != provider_type:
        msg = f"Provider alias '{alias}' is already registered"
        raise ProviderError(msg)
    _PROVIDERS[alias] = _PROVIDERS[provider_type]


def list_provider_types() -> list[str]:
    """Return all registered provider type names."""
    return sorted(_PROVIDERS)


def create_provider(
    provider_type: str,
    *,
    model_name: str,
    provider_name: str | None = None,
    runtime: ProviderRuntimeConfig | None = None,
    **config: Any,
) -> BaseModelProvider:
    """Instantiate a registered provider by type name."""
    cls = get_provider_class(provider_type)
    return cls(
        model_name=model_name,
        provider_name=provider_name or provider_type,
        runtime=runtime,
        **config,
    )


# Backward-compatible aliases
get_provider = get_provider_class
