"""Request and response types for model providers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelRequest(BaseModel):
    """A single generation request with full experiment context."""

    prompt: str
    prompt_id: str
    task_id: str
    run_id: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int | None = None
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    max_tokens: int = Field(default=1024, ge=1)
    stop: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    """Structured response from a model generation call."""

    text: str
    model_name: str
    provider_name: str
    prompt_id: str
    task_id: str
    run_id: str
    temperature: float
    seed: int | None = None
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    raw_metadata: dict[str, Any] = Field(default_factory=dict)

    def to_log_dict(self) -> dict[str, Any]:
        """Return a flat dict suitable for structured logging."""
        return self.model_dump()
