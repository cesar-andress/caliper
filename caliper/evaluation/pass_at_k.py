"""pass@k functional evaluation metrics."""

from __future__ import annotations

from caliper.evaluation.code_extraction import extract_python_code
from caliper.evaluation.context import EvaluationInput
from caliper.evaluation.sandbox import ExecutionLimits, ExecutionResult, execute_sample
from caliper.evaluation.schema import MetricEvaluationRecord


def pass_at_k(
    sample: EvaluationInput,
    *,
    k: int = 1,
    limits: ExecutionLimits | None = None,
) -> MetricEvaluationRecord:
    """Compute pass@k using sandboxed execution (k=1 for confirmatory primary metric)."""
    if k != 1:
        return MetricEvaluationRecord(
            name=f"pass_at_{k}",
            value=0.0,
            success=False,
            metadata={"reason": "only pass@1 is implemented; use k=1"},
        )

    tests = sample.tests or []
    if not tests:
        return MetricEvaluationRecord(
            name="pass_at_1",
            value=0.0,
            success=False,
            metadata={"reason": "no tests provided"},
        )

    harness = str(sample.metadata.get("harness", "generic"))
    entry_point = sample.metadata.get("entry_point")
    completion = extract_python_code(sample.prediction)
    prompt_stub = sample.metadata.get("prompt_stub") or sample.metadata.get("prompt_prefix")

    result = execute_sample(
        harness=harness,  # type: ignore[arg-type]
        prompt=prompt_stub if isinstance(prompt_stub, str) else None,
        completion=completion,
        tests=tests,
        entry_point=entry_point if isinstance(entry_point, str) else None,
        limits=limits,
    )
    return _record_from_execution(result, metric_name="pass_at_1")


def pass_at_1(sample: EvaluationInput, *, limits: ExecutionLimits | None = None) -> MetricEvaluationRecord:
    """Alias for pass@1, the confirmatory primary metric."""
    return pass_at_k(sample, k=1, limits=limits)


def execution_latency_ms(
    sample: EvaluationInput,
    *,
    limits: ExecutionLimits | None = None,
) -> MetricEvaluationRecord:
    """Secondary metric: sandbox execution latency in milliseconds."""
    record = pass_at_1(sample, limits=limits)
    latency = float(record.metadata.get("latency_ms", 0.0))
    return MetricEvaluationRecord(
        name="execution_latency_ms",
        value=latency,
        success=latency >= 0.0,
        metadata={"passed": record.success, **record.metadata},
    )


def token_count_estimate(sample: EvaluationInput) -> MetricEvaluationRecord:
    """Secondary metric: approximate completion token count from whitespace tokens."""
    text = extract_python_code(sample.prediction) or sample.prediction
    count = len(text.split())
    return MetricEvaluationRecord(
        name="token_count",
        value=float(count),
        success=True,
        metadata={"estimator": "whitespace_split"},
    )


def _record_from_execution(result: ExecutionResult, *, metric_name: str) -> MetricEvaluationRecord:
    return MetricEvaluationRecord(
        name=metric_name,
        value=1.0 if result.passed else 0.0,
        success=result.passed,
        metadata={
            "stdout": result.stdout[:2000],
            "stderr": result.stderr[:2000],
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "error_type": result.error_type,
            "error_message": result.error_message,
            "latency_ms": result.latency_ms,
            **result.metadata,
        },
    )
