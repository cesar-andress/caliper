"""Code generation benchmark task."""

from __future__ import annotations

from caliper.tasks.base import BaseTask, exact_match_score, normalize_text
from caliper.tasks.registry import register_task
from caliper.tasks.schema import TaskMetadata


@register_task("code_generation")
class CodeGenerationTask(BaseTask):
    """Evaluate code generation against expected output and optional tests."""

    domain = "code_generation"

    def score(self, example: TaskMetadata, prediction: str) -> dict[str, float]:
        scores: dict[str, float] = {
            "exact_match": exact_match_score(example.expected_output, prediction),
        }
        if example.tests:
            scores["tests_present"] = 1.0
            scores["test_pass_rate"] = self._estimate_test_pass_rate(example, prediction)
        return scores

    def _estimate_test_pass_rate(self, example: TaskMetadata, prediction: str) -> float:
        """Heuristic pass rate without executing code.

        Checks whether each test assertion string appears in the prediction
        or matches the expected output combined with the prediction.
        Real execution will replace this in a later phase.
        """
        if not example.tests:
            return 0.0
        combined = normalize_text(f"{prediction}\n{example.expected_output or ''}")
        passed = sum(1 for test in example.tests if normalize_text(test) in combined)
        return passed / len(example.tests)
