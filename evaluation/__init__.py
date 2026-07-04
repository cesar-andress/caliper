"""Evaluation metrics and post-hoc scoring."""

from caliper.evaluation.context import EvaluationInput
from caliper.evaluation.metrics import MetricResult, compute_metric
from caliper.evaluation.registry import EvaluationOptions, evaluate_sample, metrics_for_domain
from caliper.evaluation.runner import evaluate_results_dataframe, evaluate_results_file
from caliper.evaluation.schema import CellEvaluationRecord, MetricEvaluationRecord

__all__ = [
    "CellEvaluationRecord",
    "EvaluationInput",
    "EvaluationOptions",
    "MetricEvaluationRecord",
    "MetricResult",
    "compute_metric",
    "evaluate_results_dataframe",
    "evaluate_results_file",
    "evaluate_sample",
    "metrics_for_domain",
]
