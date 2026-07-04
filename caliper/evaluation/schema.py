"""Evaluation result schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class MetricEvaluationRecord(BaseModel):
    """Structured result for a single metric applied to one prediction."""

    name: str
    value: float
    success: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class CellEvaluationRecord(BaseModel):
    """Evaluation output for one experiment result row."""

    cell_id: str
    experiment_id: str
    run_id: str
    task_id: str
    domain: str
    model_id: str
    prediction: str
    expected_output: str | None = None
    metrics: list[MetricEvaluationRecord] = Field(default_factory=list)
    evaluated_at: str = Field(
        default_factory=lambda: datetime.now(tz=UTC).isoformat(),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    def metric_dict(self) -> dict[str, float]:
        return {metric.name: metric.value for metric in self.metrics}

    def primary_score(self) -> float:
        if not self.metrics:
            return 0.0
        return self.metrics[0].value
