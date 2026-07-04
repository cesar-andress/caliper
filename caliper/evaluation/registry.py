"""Metric registry and domain dispatch."""

from __future__ import annotations

from caliper.evaluation.code_metrics import contains_expected, evaluate_test_pass, exact_match
from caliper.evaluation.context import EvaluationInput
from caliper.evaluation.schema import MetricEvaluationRecord
from caliper.evaluation.summarization_metrics import length, lexical_overlap, llm_judge
from caliper.tasks.schema import TaskDomain

CODE_DOMAINS = frozenset({"code_generation", "bug_repair"})
SUMMARIZATION_DOMAINS = frozenset({"code_summarization"})


class EvaluationOptions:
    """Runtime toggles for optional / unsafe metrics."""

    def __init__(
        self,
        *,
        enable_code_execution: bool = False,
        enable_llm_judge: bool = False,
    ) -> None:
        self.enable_code_execution = enable_code_execution
        self.enable_llm_judge = enable_llm_judge


def metrics_for_domain(domain: TaskDomain) -> list[str]:
    if domain in CODE_DOMAINS:
        return ["exact_match", "contains_expected", "test_pass"]
    if domain in SUMMARIZATION_DOMAINS:
        return ["length", "lexical_overlap", "llm_judge"]
    msg = f"unsupported evaluation domain: {domain}"
    raise ValueError(msg)


def evaluate_sample(
    sample: EvaluationInput,
    *,
    options: EvaluationOptions | None = None,
) -> list[MetricEvaluationRecord]:
    """Evaluate all metrics applicable to the sample's domain."""
    opts = options or EvaluationOptions()
    if sample.domain in CODE_DOMAINS:
        return [
            exact_match(sample),
            contains_expected(sample),
            evaluate_test_pass(sample, execution_enabled=opts.enable_code_execution),
        ]
    if sample.domain in SUMMARIZATION_DOMAINS:
        return [
            length(sample),
            lexical_overlap(sample),
            llm_judge(sample, enabled=opts.enable_llm_judge),
        ]
    msg = f"unsupported evaluation domain: {sample.domain}"
    raise ValueError(msg)
