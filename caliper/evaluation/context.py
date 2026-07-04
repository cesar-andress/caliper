"""Input context for metric evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from caliper.tasks.schema import TaskDomain


@dataclass(frozen=True)
class EvaluationInput:
    """Everything a metric needs to score one prediction."""

    prediction: str
    expected_output: str | None = None
    tests: list[str] | None = None
    domain: TaskDomain = "code_generation"
    language: str = "python"
    metadata: dict[str, Any] = field(default_factory=dict)
