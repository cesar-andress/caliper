"""Bug repair benchmark task."""

from __future__ import annotations

from caliper.tasks.base import BaseTask, exact_match_score, normalize_text, token_f1_score
from caliper.tasks.registry import register_task
from caliper.tasks.schema import TaskMetadata


@register_task("bug_repair")
class BugRepairTask(BaseTask):
    """Evaluate bug-fix predictions against expected repaired code."""

    domain = "bug_repair"

    def score(self, example: TaskMetadata, prediction: str) -> dict[str, float]:
        return {
            "exact_match": exact_match_score(example.expected_output, prediction),
            "repair_f1": token_f1_score(example.expected_output, prediction),
            "bug_removed": self._bug_removed_score(example, prediction),
        }

    def _bug_removed_score(self, example: TaskMetadata, prediction: str) -> float:
        """Score whether the predicted fix differs from the buggy input."""
        if normalize_text(example.input) == normalize_text(prediction):
            return 0.0
        if example.expected_output and normalize_text(example.expected_output) == normalize_text(
            prediction
        ):
            return 1.0
        return 0.5 if prediction.strip() else 0.0
