"""Metric computation for evaluation results."""

from __future__ import annotations

from caliper.evaluation.code_metrics import contains_expected, evaluate_test_pass, exact_match
from caliper.evaluation.context import EvaluationInput
from caliper.evaluation.schema import MetricEvaluationRecord
from caliper.evaluation.summarization_metrics import length, lexical_overlap, llm_judge

# Backward-compatible alias.
MetricResult = MetricEvaluationRecord


def compute_metric(name: str, predictions: list[str], references: list[str]) -> MetricEvaluationRecord:
    """Compute a named metric over parallel prediction/reference lists.

    Supported metrics: ``accuracy``, ``exact_match``.

    Raises:
        ValueError: If the metric name is unknown or lists differ in length.
    """
    if len(predictions) != len(references):
        msg = f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        raise ValueError(msg)

    if not predictions:
        return MetricEvaluationRecord(name=name, value=0.0, success=False, metadata={"n": 0})

    if name in ("accuracy", "exact_match"):
        scores = [
            exact_match(
                EvaluationInput(prediction=pred, expected_output=ref)
            ).value
            for pred, ref in zip(predictions, references, strict=True)
        ]
        value = sum(scores) / len(scores)
        return MetricEvaluationRecord(
            name=name,
            value=value,
            success=value == 1.0,
            metadata={"n": len(predictions)},
        )

    if name == "contains_expected":
        scores = [
            contains_expected(
                EvaluationInput(prediction=pred, expected_output=ref)
            ).value
            for pred, ref in zip(predictions, references, strict=True)
        ]
        value = sum(scores) / len(scores)
        return MetricEvaluationRecord(
            name=name,
            value=value,
            success=value == 1.0,
            metadata={"n": len(predictions)},
        )

    if name == "length":
        scores = [length(EvaluationInput(prediction=pred)).value for pred in predictions]
        value = sum(scores) / len(scores)
        return MetricEvaluationRecord(
            name=name,
            value=value,
            success=all(s > 0 for s in scores),
            metadata={"n": len(predictions)},
        )

    if name == "lexical_overlap":
        scores = [
            lexical_overlap(
                EvaluationInput(prediction=pred, expected_output=ref)
            ).value
            for pred, ref in zip(predictions, references, strict=True)
        ]
        value = sum(scores) / len(scores)
        return MetricEvaluationRecord(
            name=name,
            value=value,
            success=value >= 0.5,
            metadata={"n": len(predictions)},
        )

    msg = f"Unknown metric: {name}"
    raise ValueError(msg)


__all__ = [
    "MetricEvaluationRecord",
    "MetricResult",
    "compute_metric",
    "contains_expected",
    "exact_match",
    "length",
    "lexical_overlap",
    "llm_judge",
    "evaluate_test_pass",
]
