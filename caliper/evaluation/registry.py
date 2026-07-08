"""Metric registry and domain dispatch."""

from __future__ import annotations

from caliper.evaluation.code_metrics import (
    contains_expected,
    evaluate_test_pass,
    exact_match,
    normalized_code_match,
    syntax_check,
    test_pass_rate,
)
from caliper.evaluation.context import EvaluationInput
from caliper.evaluation.schema import MetricEvaluationRecord
from caliper.evaluation.summarization_metrics import length, lexical_overlap, llm_judge
from caliper.tasks.schema import TaskDomain

CODE_DOMAINS = frozenset({"code_generation", "bug_repair", "executable_code_generation"})
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


CODE_METRIC_FUNCTIONS = {
    "exact_match": exact_match,
    "contains_expected": contains_expected,
    "normalized_code_match": normalized_code_match,
    "syntax_check": syntax_check,
    "test_pass_rate": test_pass_rate,
}


def _evaluate_executable_metric(name: str, sample: EvaluationInput) -> MetricEvaluationRecord:
    from caliper.evaluation.pass_at_k import execution_latency_ms, pass_at_1, token_count_estimate

    if name in {"pass_at_1", "pass_at_k"}:
        return pass_at_1(sample)
    if name == "execution_latency_ms":
        return execution_latency_ms(sample)
    if name == "token_count":
        return token_count_estimate(sample)
    msg = f"unsupported executable metric: {name}"
    raise ValueError(msg)


def metrics_for_domain(domain: TaskDomain) -> list[str]:
    if domain in CODE_DOMAINS:
        base = ["exact_match", "normalized_code_match", "syntax_check", "contains_expected", "test_pass"]
        if domain == "executable_code_generation":
            return ["pass_at_1", "syntax_check", "normalized_code_match", "execution_latency_ms", "token_count"]
        return base
    if domain in SUMMARIZATION_DOMAINS:
        return ["length", "lexical_overlap", "llm_judge"]
    msg = f"unsupported evaluation domain: {domain}"
    raise ValueError(msg)


def evaluate_sample(
    sample: EvaluationInput,
    *,
    options: EvaluationOptions | None = None,
    metric_names: list[str] | None = None,
) -> list[MetricEvaluationRecord]:
    """Evaluate metrics applicable to the sample's domain."""
    opts = options or EvaluationOptions()
    if sample.domain in CODE_DOMAINS:
        names = metric_names or metrics_for_domain(sample.domain)
        records: list[MetricEvaluationRecord] = []
        executable_metrics = {"pass_at_1", "pass_at_k", "execution_latency_ms", "token_count"}
        for name in names:
            if name == "test_pass":
                records.append(evaluate_test_pass(sample, execution_enabled=opts.enable_code_execution))
            elif name in CODE_METRIC_FUNCTIONS:
                records.append(CODE_METRIC_FUNCTIONS[name](sample))
            elif name in executable_metrics and sample.domain == "executable_code_generation":
                records.append(_evaluate_executable_metric(name, sample))
            else:
                msg = f"unsupported code metric: {name}"
                raise ValueError(msg)
        return records
    if sample.domain in SUMMARIZATION_DOMAINS:
        return [
            length(sample),
            lexical_overlap(sample),
            llm_judge(sample, enabled=opts.enable_llm_judge),
        ]
    msg = f"unsupported evaluation domain: {sample.domain}"
    raise ValueError(msg)
