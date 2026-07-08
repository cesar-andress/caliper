"""Code generation task scoring with extraction-aware metrics."""

from __future__ import annotations

from caliper.tasks.base import BaseTask, exact_match_score, normalize_text
from caliper.tasks.registry import register_task
from caliper.tasks.schema import TaskMetadata


@register_task("code_generation")
class CodeGenerationTask(BaseTask):
    """Evaluate code generation against expected output and optional tests."""

    domain = "code_generation"

    def score(self, example: TaskMetadata, prediction: str) -> dict[str, float]:
        from caliper.evaluation.code_metrics import (
            exact_match,
            normalized_code_match,
            syntax_check,
        )
        from caliper.evaluation.context import EvaluationInput

        sample = EvaluationInput(
            prediction=prediction,
            expected_output=example.expected_output,
            tests=example.tests,
            domain=self.domain,
            language=example.language,
        )
        scorers = {
            "exact_match": lambda: exact_match(sample).value,
            "normalized_code_match": lambda: normalized_code_match(sample).value,
            "syntax_check": lambda: syntax_check(sample).value,
            "test_pass_rate": lambda: self._estimate_test_pass_rate(example, prediction),
        }
        metric_names = self.config.get("metrics", ["exact_match"])
        scores: dict[str, float] = {}

        for metric_name in metric_names:
            if metric_name in scorers:
                scores[metric_name] = scorers[metric_name]()

        if not scores:
            scores["exact_match"] = exact_match_score(example.expected_output, prediction)

        if example.tests and "tests_present" not in scores:
            scores["tests_present"] = 1.0
            if "test_pass_rate" not in scores:
                scores["test_pass_rate"] = self._estimate_test_pass_rate(example, prediction)
        return scores

    def _estimate_test_pass_rate(self, example: TaskMetadata, prediction: str) -> float:
        """Heuristic pass rate without executing code."""
        if not example.tests:
            return 0.0
        combined = normalize_text(f"{prediction}\n{example.expected_output or ''}")
        passed = sum(1 for test in example.tests if normalize_text(test) in combined)
        return passed / len(example.tests)
