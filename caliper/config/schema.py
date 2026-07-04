"""Pydantic schemas for experiment configuration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

ProviderType = Literal["mock", "random", "openai", "anthropic", "gemini", "google", "local"]
TaskDomainType = Literal["code_generation", "bug_repair", "code_summarization"]
OutputFormat = Literal["parquet", "jsonl", "csv"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
LogFormat = Literal["json", "console"]

KNOWN_METRICS = frozenset({"accuracy", "exact_match", "f1", "pass_at_k", "bleu", "rouge_l"})
EXPERIMENT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


class DecodingConfig(BaseModel):
    """Sampling and decoding parameters applied across an experiment."""

    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    max_tokens: int = Field(default=1024, ge=1)
    stop: list[str] = Field(default_factory=list)
    seed: int | None = None


class ProviderConfig(BaseModel):
    """Named provider endpoint (API or local runtime)."""

    type: ProviderType
    api_key_env: str | None = None
    base_url: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """A model entry referencing a registered provider."""

    id: str
    provider: str
    model_id: str
    decoding: DecodingConfig | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        if not v.strip():
            msg = "model id must not be empty"
            raise ValueError(msg)
        return v


class PromptVariantConfig(BaseModel):
    """A prompt template variant for factorial prompt studies."""

    id: str
    template: str | None = None
    path: Path | None = None
    variables: dict[str, str] = Field(default_factory=dict)

    @field_validator("path", mode="before")
    @classmethod
    def _coerce_path(cls, v: str | Path | None) -> Path | None:
        return Path(v) if v is not None else None

    @model_validator(mode="after")
    def _require_template_or_path(self) -> Self:
        if self.template is None and self.path is None:
            msg = f"prompt variant '{self.id}' must specify 'template' or 'path'"
            raise ValueError(msg)
        return self


class TaskConfig(BaseModel):
    """An evaluation task / dataset definition."""

    id: str
    dataset: str
    domain: TaskDomainType | None = None
    split: str = "test"
    num_samples: int | None = Field(default=None, ge=1)
    metrics: list[str] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class OutputConfig(BaseModel):
    """Output directory and result format settings."""

    directory: Path = Field(default=Path("./outputs"))
    format: OutputFormat = "parquet"
    save_raw_responses: bool = True

    @field_validator("directory", mode="before")
    @classmethod
    def _coerce_path(cls, v: str | Path) -> Path:
        return Path(v)


class LoggingConfig(BaseModel):
    """Run-level logging configuration."""

    level: LogLevel = "INFO"
    log_to_file: bool = True
    log_format: LogFormat = "json"


class ExecutionConfig(BaseModel):
    """Execution controls for large factorial experiments."""

    shuffle: bool = True
    parallel_workers: int = Field(default=1, ge=1)


@dataclass(frozen=True)
class ExperimentCombination:
    """One cell in the experiment factorial design."""

    run_index: int
    model_id: str
    provider: str
    task_id: str
    prompt_variant_id: str
    temperature: float


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration loaded from YAML."""

    experiment_id: str
    description: str = ""
    random_seed: int = Field(default=42, ge=0)

    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    models: list[ModelConfig]
    tasks: list[TaskConfig]
    prompt_variants: list[PromptVariantConfig] = Field(default_factory=list)

    temperatures: list[float] = Field(default_factory=lambda: [0.0])
    decoding: DecodingConfig = Field(default_factory=DecodingConfig)
    evaluation_metrics: list[str] = Field(default_factory=lambda: ["accuracy"])
    number_of_runs: int = Field(default=1, ge=1)

    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("experiment_id")
    @classmethod
    def _validate_experiment_id(cls, v: str) -> str:
        if not EXPERIMENT_ID_PATTERN.match(v):
            msg = (
                "experiment_id must start with a lowercase letter and contain "
                "only lowercase letters, digits, underscores, or hyphens"
            )
            raise ValueError(msg)
        return v

    @field_validator("temperatures")
    @classmethod
    def _validate_temperatures(cls, v: list[float]) -> list[float]:
        if not v:
            msg = "temperatures must contain at least one value"
            raise ValueError(msg)
        for temp in v:
            if not 0.0 <= temp <= 2.0:
                msg = f"temperature {temp} is out of range; must be between 0.0 and 2.0"
                raise ValueError(msg)
        return v

    @field_validator("evaluation_metrics")
    @classmethod
    def _validate_metrics(cls, v: list[str]) -> list[str]:
        if not v:
            msg = "evaluation_metrics must contain at least one metric"
            raise ValueError(msg)
        cleaned = [m.strip() for m in v]
        if any(not m for m in cleaned):
            msg = "evaluation_metrics must not contain empty strings"
            raise ValueError(msg)
        return cleaned

    @model_validator(mode="after")
    def _cross_validate(self) -> Self:
        if not self.models:
            raise ValueError("models must contain at least one entry")
        if not self.tasks:
            raise ValueError("tasks must contain at least one entry")

        _check_unique_ids([m.id for m in self.models], "model")
        _check_unique_ids([t.id for t in self.tasks], "task")
        _check_unique_ids([p.id for p in self.prompt_variants], "prompt variant")

        for model in self.models:
            if model.provider not in self.providers:
                known = ", ".join(sorted(self.providers)) or "(none)"
                msg = (
                    f"model '{model.id}' references unknown provider '{model.provider}'; "
                    f"defined providers: {known}"
                )
                raise ValueError(msg)

        unknown_metrics = set(self.evaluation_metrics) - KNOWN_METRICS
        if unknown_metrics:
            known = ", ".join(sorted(KNOWN_METRICS))
            names = ", ".join(sorted(unknown_metrics))
            msg = f"unknown evaluation metric(s): {names}; known metrics: {known}"
            raise ValueError(msg)

        for task in self.tasks:
            if task.metrics:
                task_unknown = set(task.metrics) - KNOWN_METRICS
                if task_unknown:
                    names = ", ".join(sorted(task_unknown))
                    msg = f"task '{task.id}' has unknown metric(s): {names}"
                    raise ValueError(msg)

        return self

    def factorial_axes(self) -> dict[str, int]:
        """Return the size of each factorial axis."""
        return {
            "models": len(self.models),
            "temperatures": len(self.temperatures),
            "prompt_variants": max(len(self.prompt_variants), 1),
            "tasks": len(self.tasks),
            "runs": self.number_of_runs,
        }

    def total_combinations(self) -> int:
        """Total number of cells in the full factorial design."""
        total = 1
        for count in self.factorial_axes().values():
            total *= count
        return total

    def iter_combinations(self) -> Iterator[ExperimentCombination]:
        """Yield every cell in the experiment factorial design."""
        prompt_variants = self.prompt_variants or [
            PromptVariantConfig(id="default", template="{input}")
        ]
        for run_index in range(self.number_of_runs):
            for model in self.models:
                for temperature in self.temperatures:
                    for prompt in prompt_variants:
                        for task in self.tasks:
                            yield ExperimentCombination(
                                run_index=run_index,
                                model_id=model.id,
                                provider=model.provider,
                                task_id=task.id,
                                prompt_variant_id=prompt.id,
                                temperature=temperature,
                            )

    def metrics_for_task(self, task_id: str) -> list[str]:
        """Return metrics for a task, falling back to experiment-level defaults."""
        for task in self.tasks:
            if task.id == task_id:
                return task.metrics or list(self.evaluation_metrics)
        msg = f"unknown task: {task_id}"
        raise KeyError(msg)


def _check_unique_ids(ids: list[str], label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item_id in ids:
        if item_id in seen:
            duplicates.add(item_id)
        seen.add(item_id)
    if duplicates:
        names = ", ".join(sorted(duplicates))
        msg = f"duplicate {label} id(s): {names}"
        raise ValueError(msg)
