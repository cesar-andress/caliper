"""Generate human-readable experiment reports."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from caliper.config.loader import format_config_summary
from caliper.config.metrics import resolve_primary_metric
from caliper.config.schema import ExperimentConfig
from caliper.runners.reproducibility import (
    configuration_hash,
    experiment_hash,
    make_cell_hash,
)
from caliper.runners.cells import expand_cells
from caliper.statistics.descriptive import descriptive_all_factors
from caliper.statistics.prepare import prepare_results_table
from caliper.evaluation.inspect_output import metric_means_from_results


def _format_factorial_section(config: ExperimentConfig) -> str:
    axes = config.factorial_axes()
    lines = ["## Factorial dimensions", ""]
    for axis, count in axes.items():
        lines.append(f"- **{axis}**: {count}")
    lines.append("")
    lines.append(f"**Total cells**: {config.total_combinations()}")
    lines.append("")
    return "\n".join(lines)


def _format_execution_summary(manifest: dict[str, Any]) -> str:
    lines = [
        "## Execution summary",
        "",
        f"- **Status**: {manifest.get('status', 'unknown')}",
        f"- **Run ID**: `{manifest.get('run_id', '')}`",
        f"- **Started**: {manifest.get('started_at', '')}",
        f"- **Finished**: {manifest.get('finished_at', '')}",
        f"- **Duration (s)**: {manifest.get('execution_duration_seconds', 0)}",
        f"- **Total cells**: {manifest.get('total_cells', 0)}",
        f"- **Completed**: {manifest.get('completed_cells', 0)}",
        f"- **Failed**: {manifest.get('failed_cells', 0)}",
        f"- **Skipped (resumed)**: {manifest.get('skipped_cells', 0)}",
        "",
    ]
    return "\n".join(lines)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a markdown table without extra dependencies."""
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _format_metric_means(df: pd.DataFrame) -> str:
    means = metric_means_from_results(df)
    if not means:
        return "## Metric means\n\n_No metric scores available._\n"

    lines = ["## Metric means", ""]
    for name in sorted(means):
        lines.append(f"- **{name}**: {means[name]:.4f}")
    lines.append("")
    return "\n".join(lines)


def _format_output_quality_note(means: dict[str, float]) -> str:
    exact = means.get("exact_match")
    normalized = means.get("normalized_code_match")
    if exact is not None and normalized is not None and exact == 0.0 and normalized > 0.0:
        return (
            "## Output quality note\n\n"
            "Strict exact_match is zero because model outputs include Markdown or "
            "additional explanation; normalized_code_match is the primary "
            "code-equivalence metric.\n"
        )
    return ""


def _format_descriptive_stats(
    df: pd.DataFrame,
    *,
    primary_metric: str,
    warnings: list[str],
) -> str:
    if df.empty:
        return "## Descriptive statistics\n\n_No completed results to summarize._\n"

    try:
        prepared = prepare_results_table(df, metric_name=primary_metric)
    except ValueError:
        return "## Descriptive statistics\n\n_Results table could not be normalized._\n"

    factor_cols = [col for col in ("model", "task_id", "prompt_id", "temperature") if col in prepared.columns]
    tables = descriptive_all_factors(prepared, factor_cols)

    lines = [
        "## Descriptive statistics",
        "",
        f"**Primary metric**: {primary_metric}",
        "",
    ]
    for warning in warnings:
        lines.append(f"> **Warning**: {warning}")
        lines.append("")
    for factor, table in tables.items():
        lines.append(f"### By `{factor}`")
        lines.append("")
        lines.append(_dataframe_to_markdown(table))
        lines.append("")
    return "\n".join(lines)


def _format_failures(df: pd.DataFrame) -> str:
    if df.empty or "status" not in df.columns:
        return "## Failures\n\n_No failure records._\n"

    failed = df[df["status"] == "failed"]
    if failed.empty:
        return "## Failures\n\n_No failed cells._\n"

    lines = ["## Failures", ""]
    for row in failed.to_dict(orient="records"):
        lines.append(
            f"- `{row.get('cell_id', '')}` "
            f"(model={row.get('model_id')}, task={row.get('task_id')}, "
            f"run={row.get('run_index')}): {row.get('error', 'unknown error')}"
        )
    lines.append("")
    return "\n".join(lines)


def _format_hardware(manifest: dict[str, Any]) -> str:
    cpu = manifest.get("cpu", {})
    gpu = manifest.get("gpu", {})
    lines = [
        "## Hardware",
        "",
        f"- **CPU**: {cpu.get('processor', 'unknown')} ({cpu.get('machine', '')})",
        f"- **Platform**: {cpu.get('platform', '')}",
        f"- **GPU available**: {gpu.get('gpu_available', False)}",
    ]
    if gpu.get("gpu_name"):
        lines.append(f"- **GPU**: {gpu['gpu_name']}")
    lines.append("")
    return "\n".join(lines)


def _format_reproducibility(config: ExperimentConfig, manifest: dict[str, Any]) -> str:
    cells = expand_cells(config)
    cell_hashes = [make_cell_hash(config, cell) for cell in cells]
    lines = [
        "## Reproducibility",
        "",
        f"- **Configuration hash**: `{manifest.get('configuration_hash', configuration_hash(config))}`",
        f"- **Experiment hash**: `{manifest.get('experiment_hash', experiment_hash(config, cell_hashes))}`",
        f"- **Random seed**: {manifest.get('random_seed', config.random_seed)}",
        f"- **Git commit**: `{manifest.get('git_commit', 'unknown')}`",
        f"- **Software version**: {manifest.get('software_version', 'unknown')}",
        f"- **Python**: {manifest.get('python_version', '')}",
        "",
        "### Libraries",
        "",
    ]
    for name, version in sorted(manifest.get("libraries", {}).items()):
        lines.append(f"- `{name}`: {version}")
    lines.append("")
    return "\n".join(lines)


def generate_report(
    *,
    config: ExperimentConfig,
    manifest: dict[str, Any],
    results_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Write ``report.md`` for a completed experiment."""
    generated_at = datetime.now().isoformat()
    primary_metric, primary_warnings = resolve_primary_metric(config)
    metric_means = metric_means_from_results(results_df)
    sections = [
        f"# Experiment report: {config.experiment_id}",
        "",
        f"_Generated at {generated_at}_",
        "",
        "## Configuration summary",
        "",
        "```",
        format_config_summary(config),
        "```",
        "",
        _format_factorial_section(config),
        _format_execution_summary(manifest),
        _format_metric_means(results_df),
        _format_output_quality_note(metric_means),
        _format_descriptive_stats(
            results_df,
            primary_metric=primary_metric,
            warnings=primary_warnings,
        ),
        _format_failures(results_df),
        _format_hardware(manifest),
        _format_reproducibility(config, manifest),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")
    return output_path
