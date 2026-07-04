"""Metrics for code summarization tasks."""

from __future__ import annotations

from caliper.evaluation.context import EvaluationInput
from caliper.evaluation.schema import MetricEvaluationRecord
from caliper.evaluation.text_utils import tokenize


def length(sample: EvaluationInput) -> MetricEvaluationRecord:
    """Report prediction length in words and characters."""
    words = tokenize(sample.prediction)
    word_count = len(words)
    char_count = len(sample.prediction)
    return MetricEvaluationRecord(
        name="length",
        value=float(word_count),
        success=word_count > 0,
        metadata={
            "word_count": word_count,
            "char_count": char_count,
        },
    )


def lexical_overlap(sample: EvaluationInput) -> MetricEvaluationRecord:
    """Simple token-level F1 overlap between prediction and reference."""
    if sample.expected_output is None:
        return MetricEvaluationRecord(
            name="lexical_overlap",
            value=0.0,
            success=False,
            metadata={"reason": "no expected_output provided"},
        )

    ref_tokens = set(tokenize(sample.expected_output))
    pred_tokens = set(tokenize(sample.prediction))
    if not ref_tokens or not pred_tokens:
        return MetricEvaluationRecord(
            name="lexical_overlap",
            value=0.0,
            success=False,
            metadata={"ref_token_count": len(ref_tokens), "pred_token_count": len(pred_tokens)},
        )

    overlap = ref_tokens & pred_tokens
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(ref_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    success = f1 >= 0.5

    return MetricEvaluationRecord(
        name="lexical_overlap",
        value=f1,
        success=success,
        metadata={
            "precision": precision,
            "recall": recall,
            "overlap_tokens": len(overlap),
            "threshold": 0.5,
        },
    )


def llm_judge(
    sample: EvaluationInput,
    *,
    enabled: bool = False,
) -> MetricEvaluationRecord:
    """Placeholder for LLM-as-judge evaluation.

    Disabled by default; returns a structured skipped record until a judge
    provider is configured.
    """
    if not enabled:
        return MetricEvaluationRecord(
            name="llm_judge",
            value=0.0,
            success=False,
            metadata={
                "status": "disabled",
                "reason": "LLM-as-judge disabled by default",
            },
        )

    return MetricEvaluationRecord(
        name="llm_judge",
        value=0.0,
        success=False,
        metadata={
            "status": "not_implemented",
            "reason": "LLM-as-judge provider not yet implemented",
        },
    )
