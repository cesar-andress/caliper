"""Post-execution pipeline: evaluate, statistical dataset, manifest, report."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import structlog
import yaml

from caliper.config.schema import ExperimentConfig
from caliper.config.metrics import resolve_primary_metric
from caliper.runners.artifact_export import export_artifact
from caliper.runners.manifest_builder import build_manifest, write_experiment_manifest
from caliper.runners.report import generate_report
from caliper.statistics.prepare import prepare_results_table
from caliper.storage.formats import read_results, write_results

logger = structlog.get_logger(__name__)


def ensure_output_layout(output_dir: Path) -> dict[str, Path]:
    """Create the standard experiment output directory layout."""
    logs_dir = output_dir / "logs"
    figures_dir = output_dir / "figures"
    checkpoints_dir = output_dir / "checkpoints"
    for path in (logs_dir, figures_dir, checkpoints_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "logs": logs_dir,
        "figures": figures_dir,
        "checkpoints": checkpoints_dir,
    }


def copy_config_snapshot(config_path: Path | None, output_dir: Path, config: ExperimentConfig) -> Path:
    """Persist ``config.yaml`` in the experiment output directory."""
    target = output_dir / "config.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    if config_path is not None and config_path.exists():
        shutil.copy2(config_path, target)
    else:
        target.write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
    return target


def build_statistical_dataset(
    results_df: pd.DataFrame,
    output_dir: Path,
    *,
    primary_metric: str | None = None,
) -> Path | None:
    """Normalize results into the Paper 1 statistical schema."""
    completed = results_df[results_df["status"] == "completed"].copy() if not results_df.empty else results_df
    if completed.empty:
        return None

    prepared = prepare_results_table(completed, metric_name=primary_metric)
    path = output_dir / "statistical_dataset.parquet"
    write_results(prepared, path, fmt="parquet")
    return path


def finalize_experiment(
    *,
    config: ExperimentConfig,
    config_path: Path | None,
    output_dir: Path,
    run_id: str,
    started_at: datetime,
    finished_at: datetime,
    total_cells: int,
    completed_cells: int,
    failed_cells: int,
    skipped_cells: int,
    status: str,
    execution_duration_seconds: float,
    result_paths: dict[str, Path] | None = None,
    evaluate: bool = True,
) -> dict[str, Any]:
    """Run post-execution steps and write manifest/report artifacts."""
    layout = ensure_output_layout(output_dir)
    copy_config_snapshot(config_path, output_dir, config)

    parquet_path = output_dir / "results.parquet"
    results_df = read_results(parquet_path) if parquet_path.exists() else pd.DataFrame()

    eval_summary: dict[str, Any] = {}
    if evaluate and parquet_path.exists():
        from caliper.evaluation.runner import evaluate_results_file

        eval_summary = evaluate_results_file(
            parquet_path,
            config,
            config_path=config_path,
            output_dir=output_dir,
        )

    stats_path = build_statistical_dataset(
        results_df,
        output_dir,
        primary_metric=resolve_primary_metric(config)[0],
    )

    str_paths = {k: str(v) for k, v in (result_paths or {}).items()}
    if stats_path is not None:
        str_paths["statistical_dataset"] = str(stats_path)

    manifest = build_manifest(
        config=config,
        run_id=run_id,
        output_dir=output_dir,
        config_path=config_path,
        started_at=started_at,
        finished_at=finished_at,
        total_cells=total_cells,
        completed_cells=completed_cells,
        failed_cells=failed_cells,
        skipped_cells=skipped_cells,
        status=status,
        execution_duration_seconds=execution_duration_seconds,
        random_seed=config.random_seed,
        result_paths=str_paths,
    )
    manifest["evaluation"] = eval_summary
    manifest["directories"] = {k: str(v) for k, v in layout.items()}

    write_experiment_manifest(output_dir, manifest)
    generate_report(
        config=config,
        manifest=manifest,
        results_df=results_df,
        output_path=output_dir / "report.md",
    )

    logger.info(
        "pipeline.finalized",
        output_dir=str(output_dir),
        manifest=str(output_dir / "manifest.json"),
        report=str(output_dir / "report.md"),
        statistical_dataset=str(stats_path) if stats_path else None,
    )

    if status == "completed" and parquet_path.exists():
        artifact_result = export_artifact(output_dir)
        manifest["artifact"] = {
            "path": str(artifact_result.artifact_dir),
            "complete": artifact_result.verification.complete,
            "warnings": artifact_result.verification.warnings,
            "errors": artifact_result.verification.errors,
            "missing_files": artifact_result.verification.missing_files,
        }
        write_experiment_manifest(output_dir, manifest)

    return manifest
