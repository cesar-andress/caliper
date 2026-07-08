"""Primary metric resolution for reports and analysis."""

from __future__ import annotations

from pathlib import Path

from caliper.config.loader import load_config
from caliper.config.schema import ExperimentConfig


def configured_metrics(config: ExperimentConfig) -> set[str]:
    """Return all metric names referenced in the experiment config."""
    names = set(config.evaluation_metrics)
    for task in config.tasks:
        if task.metrics:
            names.update(task.metrics)
    return names


def resolve_primary_metric(config: ExperimentConfig) -> tuple[str, list[str]]:
    """Resolve the primary metric for reporting and analysis.

    Returns:
        Tuple of (metric name, warning messages).
    """
    warnings: list[str] = []
    if config.primary_metric is not None:
        return config.primary_metric, warnings

    metric = config.evaluation_metrics[0]
    warnings.append(
        "primary_metric not configured; defaulting to the first evaluation metric "
        f"({metric}). Set primary_metric explicitly for publication-ready reports."
    )
    return metric, warnings


def load_config_for_results(
    results_path: Path,
    config_path: Path | None = None,
) -> ExperimentConfig | None:
    """Load experiment config from an explicit path or the results directory."""
    if config_path is not None:
        return load_config(config_path)

    candidate = results_path.parent / "config.yaml"
    if candidate.exists():
        return load_config(candidate)
    return None


def resolve_analysis_metric(
    *,
    metric: str | None,
    results_path: Path,
    config_path: Path | None = None,
) -> tuple[str | None, list[str]]:
    """Resolve metric for CLI/scripts: explicit flag, else config primary."""
    if metric is not None:
        return metric, []

    config = load_config_for_results(results_path, config_path)
    if config is None:
        return None, [
            "no --metric provided and no experiment config found; "
            "using the score column as stored in results."
        ]

    return resolve_primary_metric(config)
