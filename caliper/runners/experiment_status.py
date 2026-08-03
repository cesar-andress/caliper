"""Lightweight progress reporting for long-running factorial experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from caliper.config.loader import load_config
from caliper.runners.checkpoint import CheckpointStore
from caliper.runners.experiment import ExperimentRunner
from caliper.runners.failures import count_terminal_failures
from caliper.runners.missing_cells import inspect_missing_cells
from caliper.runners.experiment_paths import resolve_experiment_dir


@dataclass(frozen=True)
class ExperimentStatus:
    experiment_id: str
    experiment_dir: Path
    expected_cells: int
    completed_cells: int
    failed_cells: int
    skipped_cells: int
    pending_cells: int
    percent_complete: float
    elapsed_seconds: float | None
    throughput_cells_per_hour: float | None
    eta_seconds: float | None
    pass_at_1_mean: float | None
    model_distribution: dict[str, int]
    run_id: str | None
    status: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_dir": str(self.experiment_dir),
            "expected_cells": self.expected_cells,
            "completed_cells": self.completed_cells,
            "failed_cells": self.failed_cells,
            "skipped_cells": self.skipped_cells,
            "pending_cells": self.pending_cells,
            "percent_complete": self.percent_complete,
            "elapsed_seconds": self.elapsed_seconds,
            "throughput_cells_per_hour": self.throughput_cells_per_hour,
            "eta_seconds": self.eta_seconds,
            "pass_at_1_mean": self.pass_at_1_mean,
            "model_distribution": self.model_distribution,
            "run_id": self.run_id,
            "status": self.status,
        }


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _metric_mean(experiment_dir: Path, metric: str = "pass_at_1") -> float | None:
    stats_path = experiment_dir / "statistical_dataset.parquet"
    results_path = experiment_dir / "results.parquet"
    path = stats_path if stats_path.exists() else results_path
    if not path.exists():
        return None
    try:
        import pandas as pd

        frame = pd.read_parquet(path)
    except Exception:
        return None
    if metric in frame.columns:
        series = frame[metric]
    elif "metric_name" in frame.columns and "metric_value" in frame.columns:
        series = frame.loc[frame["metric_name"] == metric, "metric_value"]
    elif "metric_value" in frame.columns:
        series = frame["metric_value"]
    else:
        return None
    if series.empty:
        return None
    return float(series.mean())


def _model_distribution(experiment_dir: Path) -> dict[str, int]:
    results_path = experiment_dir / "results.parquet"
    if not results_path.exists():
        return {}
    try:
        import pandas as pd

        frame = pd.read_parquet(results_path)
    except Exception:
        return {}
    if "model_id" in frame.columns:
        col = "model_id"
    elif "model" in frame.columns:
        col = "model"
    else:
        return {}
    counts = frame[col].value_counts().to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def collect_experiment_status(
    experiment_dir: Path | str,
    *,
    config_path: Path | str | None = None,
) -> ExperimentStatus:
    """Summarize completion progress for a factorial experiment directory."""
    experiment_dir = resolve_experiment_dir(experiment_dir)
    manifest_path = experiment_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    config_file = Path(config_path) if config_path else experiment_dir / "config.yaml"
    if not config_file.exists():
        config_file = experiment_dir.parent / "config.yaml"
    experiment_config = load_config(config_file) if config_file.exists() else None

    expected = int(manifest.get("total_cells") or 0)
    if expected <= 0 and experiment_config is not None:
        expected = len(ExperimentRunner(experiment_config, dry_run=True).plan_combinations())

    checkpoint_store = CheckpointStore(experiment_dir / "checkpoints")
    completed = len(checkpoint_store.load_completed_cell_ids())
    failed = len(checkpoint_store.load_failed_cell_ids())
    failures_path = experiment_dir / "failures.jsonl"
    if failures_path.exists():
        failed = max(failed, count_terminal_failures(failures_path))

    skipped = int(manifest.get("skipped_cells") or 0)
    pending = max(expected - completed - failed, 0)
    percent = (100.0 * completed / expected) if expected else 0.0

    started = _parse_timestamp(manifest.get("started_at"))
    finished = _parse_timestamp(manifest.get("finished_at"))
    now = finished or datetime.now(tz=UTC)
    elapsed = (now - started).total_seconds() if started else None
    throughput = None
    eta = None
    if elapsed and elapsed > 0 and completed > 0:
        throughput = completed / (elapsed / 3600.0)
        if pending > 0:
            eta = pending / (completed / elapsed)

    return ExperimentStatus(
        experiment_id=str(manifest.get("experiment_id") or (experiment_config.experiment_id if experiment_config else experiment_dir.name)),
        experiment_dir=experiment_dir,
        expected_cells=expected,
        completed_cells=completed,
        failed_cells=failed,
        skipped_cells=skipped,
        pending_cells=pending,
        percent_complete=percent,
        elapsed_seconds=elapsed,
        throughput_cells_per_hour=throughput,
        eta_seconds=eta,
        pass_at_1_mean=_metric_mean(experiment_dir),
        model_distribution=_model_distribution(experiment_dir),
        run_id=manifest.get("run_id"),
        status=manifest.get("status"),
    )


def format_experiment_status(status: ExperimentStatus) -> str:
    """Render a human-readable status report."""
    lines = [
        f"Experiment: {status.experiment_id}",
        f"Directory:  {status.experiment_dir}",
        f"Status:     {status.status or 'unknown'} (run {status.run_id or 'n/a'})",
        "",
        f"Expected:   {status.expected_cells:,}",
        f"Completed:  {status.completed_cells:,}",
        f"Failed:     {status.failed_cells:,}",
        f"Skipped:    {status.skipped_cells:,}",
        f"Pending:    {status.pending_cells:,}",
        f"Progress:   {status.percent_complete:.2f}%",
    ]
    if status.elapsed_seconds is not None:
        hours = status.elapsed_seconds / 3600.0
        lines.append(f"Elapsed:    {hours:.2f} h")
    if status.throughput_cells_per_hour is not None:
        lines.append(f"Throughput: {status.throughput_cells_per_hour:.1f} cells/h")
    if status.eta_seconds is not None:
        lines.append(f"ETA:        {status.eta_seconds / 3600.0:.2f} h")
    if status.pass_at_1_mean is not None:
        lines.append(f"Pass@1:     {status.pass_at_1_mean:.4f} (completed cells)")
    if status.model_distribution:
        lines.append("")
        lines.append("Completed cells by model:")
        for model, count in sorted(status.model_distribution.items()):
            lines.append(f"  - {model}: {count}")
    return "\n".join(lines)
