"""Metrics for code generation and bug repair tasks."""

from __future__ import annotations

from caliper.evaluation.context import EvaluationInput
from caliper.evaluation.schema import MetricEvaluationRecord
from caliper.evaluation.text_utils import normalize_text


def exact_match(sample: EvaluationInput) -> MetricEvaluationRecord:
    """Return 1.0 when prediction exactly matches expected output."""
    if sample.expected_output is None:
        return MetricEvaluationRecord(
            name="exact_match",
            value=0.0,
            success=False,
            metadata={"reason": "no expected_output provided"},
        )

    matched = normalize_text(sample.prediction) == normalize_text(sample.expected_output)
    return MetricEvaluationRecord(
        name="exact_match",
        value=float(matched),
        success=matched,
        metadata={
            "prediction_length": len(sample.prediction),
            "reference_length": len(sample.expected_output),
        },
    )


def contains_expected(sample: EvaluationInput) -> MetricEvaluationRecord:
    """Return 1.0 when expected output is contained in the prediction."""
    if sample.expected_output is None:
        return MetricEvaluationRecord(
            name="contains_expected",
            value=0.0,
            success=False,
            metadata={"reason": "no expected_output provided"},
        )

    expected = normalize_text(sample.expected_output)
    prediction = normalize_text(sample.prediction)
    contained = expected in prediction if expected else False
    return MetricEvaluationRecord(
        name="contains_expected",
        value=float(contained),
        success=contained,
        metadata={"match_type": "substring"},
    )


def evaluate_test_pass(
    sample: EvaluationInput,
    *,
    execution_enabled: bool = False,
) -> MetricEvaluationRecord:
    """Placeholder for sandboxed test execution.

    Code execution is disabled by default for safety. When disabled, returns
    a structured record indicating the metric was skipped.
    """
    if not execution_enabled:
        return MetricEvaluationRecord(
            name="test_pass",
            value=0.0,
            success=False,
            metadata={
                "status": "disabled",
                "reason": "code execution disabled by default",
                "test_count": len(sample.tests or []),
            },
        )

    if not sample.tests:
        return MetricEvaluationRecord(
            name="test_pass",
            value=0.0,
            success=False,
            metadata={"status": "skipped", "reason": "no tests provided"},
        )

    return MetricEvaluationRecord(
        name="test_pass",
        value=0.0,
        success=False,
        metadata={
            "status": "not_implemented",
            "reason": "sandboxed execution not yet implemented",
            "test_count": len(sample.tests),
        },
    )
