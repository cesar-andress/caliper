"""Abstract base class and shared scoring utilities for benchmark tasks."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from caliper.tasks.loader import TaskDataset
from caliper.tasks.schema import TaskDomain, TaskMetadata
from caliper.tasks.validation import TaskValidationError, validate_dataset


@dataclass
class TaskResult:
    """Result of evaluating a model response on one task instance."""

    task_id: str
    model_name: str
    prompt_id: str
    run_index: int
    prediction: str
    score: float
    metric: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTask(ABC):
    """Abstract interface for a benchmark evaluation task."""

    domain: TaskDomain

    def __init__(
        self,
        task_id: str,
        dataset: TaskDataset | Path | str,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.task_id = task_id
        self.config = config or {}
        if isinstance(dataset, (str, Path)):
            self.dataset = TaskDataset.from_jsonl(dataset)
        else:
            self.dataset = dataset
        self._validate()

    def _validate(self) -> None:
        errors = validate_dataset(self.dataset, expected_domain=self.domain)
        if errors:
            raise TaskValidationError(errors)

    def load_examples(self) -> list[TaskMetadata]:
        """Return all task instances in this dataset."""
        records = self.dataset.records
        filter_task_id = self.config.get("filter_task_id")
        if filter_task_id:
            records = [record for record in records if record.task_id == filter_task_id]
        limit = self.config.get("num_samples")
        if limit is not None:
            return records[: int(limit)]
        return list(records)

    @abstractmethod
    def score(self, example: TaskMetadata, prediction: str) -> dict[str, float]:
        """Score a model prediction against one task instance."""

    def num_examples(self) -> int:
        return len(self.load_examples())


def normalize_text(text: str) -> str:
    """Normalize whitespace for comparison."""
    return re.sub(r"\s+", " ", text.strip())


def exact_match_score(reference: str | None, prediction: str) -> float:
    if reference is None:
        return 0.0
    return float(normalize_text(reference) == normalize_text(prediction))


def token_f1_score(reference: str | None, prediction: str) -> float:
    if reference is None:
        return 0.0
    ref_tokens = set(normalize_text(reference).lower().split())
    pred_tokens = set(normalize_text(prediction).lower().split())
    if not ref_tokens or not pred_tokens:
        return 0.0
    overlap = ref_tokens & pred_tokens
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
