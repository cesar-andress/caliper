"""Metrics for code generation and bug repair tasks."""

from __future__ import annotations

import ast

from caliper.evaluation.code_extraction import extract_python_code, normalize_code
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


def normalized_code_match(sample: EvaluationInput) -> MetricEvaluationRecord:
    """Compare extracted, normalized Python code against the reference."""
    if sample.expected_output is None:
        return MetricEvaluationRecord(
            name="normalized_code_match",
            value=0.0,
            success=False,
            metadata={"reason": "no expected_output provided"},
        )

    predicted = normalize_code(sample.prediction)
    expected = normalize_code(sample.expected_output)
    matched = predicted == expected and bool(expected)
    return MetricEvaluationRecord(
        name="normalized_code_match",
        value=float(matched),
        success=matched,
        metadata={
            "extracted_prediction": extract_python_code(sample.prediction),
            "prediction_length": len(predicted),
            "reference_length": len(expected),
        },
    )


def syntax_check(sample: EvaluationInput) -> MetricEvaluationRecord:
    """Return 1.0 when extracted Python code parses with ``ast.parse``."""
    extracted = extract_python_code(sample.prediction)
    if not extracted.strip():
        return MetricEvaluationRecord(
            name="syntax_check",
            value=0.0,
            success=False,
            metadata={"reason": "empty prediction"},
        )

    try:
        ast.parse(extracted)
    except SyntaxError as exc:
        return MetricEvaluationRecord(
            name="syntax_check",
            value=0.0,
            success=False,
            metadata={
                "extracted_code": extracted,
                "syntax_error": str(exc),
                "lineno": exc.lineno,
            },
        )

    return MetricEvaluationRecord(
        name="syntax_check",
        value=1.0,
        success=True,
        metadata={"extracted_code": extracted},
    )


def test_pass_rate(sample: EvaluationInput) -> MetricEvaluationRecord:
    """Heuristic pass rate via substring match (no code execution)."""
    tests = sample.tests or []
    if not tests:
        return MetricEvaluationRecord(
            name="test_pass_rate",
            value=0.0,
            success=False,
            metadata={"reason": "no tests provided"},
        )

    combined = normalize_text(f"{sample.prediction}\n{sample.expected_output or ''}")
    passed = sum(1 for test in tests if normalize_text(test) in combined)
    rate = passed / len(tests)
    return MetricEvaluationRecord(
        name="test_pass_rate",
        value=rate,
        success=rate >= 1.0,
        metadata={"tests_present": len(tests), "tests_matched": passed},
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

    from caliper.evaluation.pass_at_k import pass_at_1

    record = pass_at_1(sample)
    return MetricEvaluationRecord(
        name="test_pass",
        value=record.value,
        success=record.success,
        metadata={"status": "executed", **record.metadata},
    )
