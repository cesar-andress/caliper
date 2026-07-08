"""Executable code generation with sandboxed pass@1 evaluation."""

from __future__ import annotations

from typing import Any

from caliper.tasks.base import BaseTask
from caliper.tasks.registry import register_task
from caliper.tasks.schema import TaskMetadata


@register_task("executable_code_generation")
class ExecutableCodeGenerationTask(BaseTask):
    """Evaluate code generation with real unit-test execution (pass@1)."""

    domain = "executable_code_generation"

    def score(self, example: TaskMetadata, prediction: str) -> dict[str, float]:
        from caliper.evaluation.code_metrics import normalized_code_match, syntax_check
        from caliper.evaluation.context import EvaluationInput
        from caliper.evaluation.pass_at_k import (
            execution_latency_ms,
            pass_at_1,
            token_count_estimate,
        )
        from caliper.evaluation.sandbox import ExecutionLimits

        limits = self._execution_limits()
        sample = EvaluationInput(
            prediction=prediction,
            expected_output=example.expected_output,
            tests=example.tests,
            domain=self.domain,
            language=example.language,
            metadata={
                **example.extra,
                "prompt_stub": example.input,
            },
        )

        scorers: dict[str, Any] = {
            "pass_at_1": lambda: pass_at_1(sample, limits=limits),
            "pass_at_k": lambda: pass_at_1(sample, limits=limits),
            "syntax_check": lambda: syntax_check(sample),
            "normalized_code_match": lambda: normalized_code_match(sample),
            "execution_latency_ms": lambda: execution_latency_ms(sample, limits=limits),
            "token_count": lambda: token_count_estimate(sample),
        }

        metric_names = self.config.get("metrics", ["pass_at_1"])
        scores: dict[str, float] = {}
        for metric_name in metric_names:
            if metric_name in scorers:
                scores[metric_name] = float(scorers[metric_name]().value)

        if not scores:
            scores["pass_at_1"] = float(pass_at_1(sample, limits=limits).value)
        return scores

    def _execution_limits(self) -> ExecutionLimits:
        from caliper.evaluation.sandbox import ExecutionLimits

        cfg = self.config.get("execution", {})
        return ExecutionLimits(
            timeout_seconds=float(cfg.get("timeout_seconds", 5.0)),
            memory_mb=int(cfg.get("memory_mb", 512)),
        )
