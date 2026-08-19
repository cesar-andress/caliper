"""Request and response types for model providers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ThinkMode = Literal["auto", "true", "false"] | bool | None


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
    # Reasoning-model control for Ollama Qwen3-class models.
    # auto: omit think (provider default); true/false: send top-level think.
    think: ThinkMode = "auto"
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
    done_reason: str | None = None
    thinking: str | None = None
    thinking_length: int = 0
    thinking_sha256: str | None = None
    budget_exhausted: bool = False
    raw_metadata: dict[str, Any] = Field(default_factory=dict)

    def to_log_dict(self) -> dict[str, Any]:
        """Return a flat dict suitable for structured logging."""
        dump = self.model_dump()
        # Avoid flooding logs with full thinking / raw payloads.
        if dump.get("thinking") and len(dump["thinking"]) > 200:
            dump["thinking"] = dump["thinking"][:200] + "…"
        raw = dump.get("raw_metadata") or {}
        if "ollama" in raw and isinstance(raw["ollama"], dict):
            ollama = dict(raw["ollama"])
            if isinstance(ollama.get("thinking"), str) and len(ollama["thinking"]) > 200:
                ollama["thinking"] = ollama["thinking"][:200] + "…"
            if isinstance(ollama.get("context"), list):
                ollama["context"] = f"<{len(ollama['context'])} tokens omitted>"
            raw = {**raw, "ollama": ollama}
            dump["raw_metadata"] = raw
        return dump
