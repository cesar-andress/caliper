"""Inspect completed experiment outputs and metric values."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from caliper.config.loader import load_config
from caliper.storage.formats import read_results


def _load_expected_outputs(experiment_dir: Path) -> dict[str, str]:
    eval_path = experiment_dir / "evaluations.parquet"
    if not eval_path.exists():
        return {}

    eval_df = read_results(eval_path)
    if eval_df.empty or "cell_id" not in eval_df.columns:
        return {}

    mapping: dict[str, str] = {}
    for cell_id, group in eval_df.groupby("cell_id", sort=False):
        expected = group["expected_output"].iloc[0] if "expected_output" in group.columns else ""
        mapping[str(cell_id)] = str(expected) if expected is not None else ""
    return mapping


def _metrics_for_cell(eval_df: pd.DataFrame | None, cell_id: str, scores: Any) -> dict[str, float]:
    if eval_df is not None and not eval_df.empty:
        cell_metrics = eval_df[eval_df["cell_id"] == cell_id]
        if not cell_metrics.empty and "metric_name" in cell_metrics.columns:
            return {
                str(row["metric_name"]): float(row["metric_value"])
                for row in cell_metrics.to_dict(orient="records")
                if row.get("metric_value") is not None
            }

    if isinstance(scores, dict):
        return {
            str(key): float(value)
            for key, value in scores.items()
            if value is not None
        }
    return {}


def inspect_experiment(experiment_dir: Path, *, limit: int = 5) -> list[dict[str, Any]]:
    """Load and return inspection records for an experiment directory."""
    experiment_dir = experiment_dir.resolve()
    results_path = experiment_dir / "results.parquet"
    if not results_path.exists():
        msg = f"results file not found: {results_path}"
        raise FileNotFoundError(msg)

    df = read_results(results_path)
    if "status" in df.columns:
        df = df[df["status"] == "completed"]
    sample = df.head(limit)

    eval_df = None
    eval_path = experiment_dir / "evaluations.parquet"
    if eval_path.exists():
        eval_df = read_results(eval_path)

    expected_by_cell = _load_expected_outputs(experiment_dir)
    records: list[dict[str, Any]] = []

    for row in sample.to_dict(orient="records"):
        cell_id = str(row.get("cell_id", ""))
        expected = expected_by_cell.get(cell_id, "")
        if not expected and eval_df is not None:
            matches = eval_df[eval_df["cell_id"] == cell_id]
            if not matches.empty and "expected_output" in matches.columns:
                expected = str(matches["expected_output"].iloc[0] or "")

        metrics = _metrics_for_cell(eval_df, cell_id, row.get("scores"))
        records.append(
            {
                "cell_id": cell_id,
                "task_id": row.get("task_id"),
                "prompt_id": row.get("prompt_variant_id"),
                "model": row.get("model_id"),
                "prediction": str(row.get("prediction", "")),
                "expected_output": expected,
                "metrics": metrics,
            }
        )

    return records


def format_inspection(records: list[dict[str, Any]]) -> str:
    """Format inspection records for CLI output."""
    if not records:
        return "No completed rows to inspect."

    lines: list[str] = []
    for index, record in enumerate(records, start=1):
        lines.extend(
            [
                f"=== Sample {index} ===",
                f"task_id: {record.get('task_id')}",
                f"prompt_id: {record.get('prompt_id')}",
                f"model: {record.get('model')}",
                "",
                "prediction:",
                str(record.get("prediction", "")),
                "",
                "expected_output:",
                str(record.get("expected_output", "")),
                "",
                "metrics:",
            ]
        )
        metrics = record.get("metrics") or {}
        if metrics:
            for name, value in sorted(metrics.items()):
                lines.append(f"  {name}: {value}")
        else:
            lines.append("  (none)")
        lines.append("")

    return "\n".join(lines).rstrip()


def metric_means_from_results(df: pd.DataFrame) -> dict[str, float]:
    """Compute mean values for each metric in a results table."""
    if df.empty:
        return {}

    completed = df[df["status"] == "completed"] if "status" in df.columns else df
    totals: dict[str, list[float]] = {}

    if "scores" in completed.columns:
        for scores in completed["scores"]:
            if isinstance(scores, dict):
                for name, value in scores.items():
                    if value is not None:
                        totals.setdefault(str(name), []).append(float(value))

    if totals:
        return {
            name: sum(values) / len(values)
            for name, values in totals.items()
            if values
        }

    metric_col = "metric" if "metric" in completed.columns else None
    if metric_col and "score" in completed.columns:
        for metric_name, group in completed.groupby(metric_col, observed=True):
            totals[str(metric_name)] = [
                float(value) for value in group["score"] if value is not None
            ]
        return {
            name: sum(values) / len(values)
            for name, values in totals.items()
            if values
        }

    return {}


def load_experiment_config(experiment_dir: Path):
    config_path = experiment_dir / "config.yaml"
    if config_path.exists():
        return load_config(config_path)
    return None
