"""Build publication-ready experiment manifests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from caliper.config.schema import ExperimentConfig
from caliper.runners.reproducibility import (
    collect_environment,
    configuration_hash,
    experiment_hash,
    make_cell_hash,
)
from caliper.runners.cells import expand_cells
from caliper.storage.formats import write_manifest


def build_manifest(
    *,
    config: ExperimentConfig,
    run_id: str,
    output_dir: Path,
    config_path: Path | None,
    started_at: datetime,
    finished_at: datetime | None,
    total_cells: int,
    completed_cells: int,
    failed_cells: int,
    skipped_cells: int,
    status: str,
    execution_duration_seconds: float,
    random_seed: int,
    result_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble a reproducibility manifest for an experiment run."""
    cells = expand_cells(config)
    cell_hashes = [make_cell_hash(config, cell) for cell in cells]
    env = collect_environment()

    return {
        "run_id": run_id,
        "experiment_id": config.experiment_id,
        "status": status,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat() if finished_at else None,
        "execution_duration_seconds": round(execution_duration_seconds, 3),
        "config_path": str(config_path) if config_path else None,
        "output_dir": str(output_dir),
        "random_seed": random_seed,
        "configuration_hash": configuration_hash(config),
        "experiment_hash": experiment_hash(config, cell_hashes),
        "total_cells": total_cells,
        "completed_cells": completed_cells,
        "failed_cells": failed_cells,
        "skipped_cells": skipped_cells,
        "software_version": env["software_version"],
        "git_commit": env["git_commit"],
        "python_version": env["python_version"],
        "os": env["os"],
        "cpu": env["cpu"],
        "gpu": env["gpu"],
        "libraries": env["libraries"],
        "factorial_axes": config.factorial_axes(),
        "result_paths": result_paths or {},
    }


def write_experiment_manifest(output_dir: Path, manifest: dict[str, Any]) -> Path:
    """Write ``manifest.json`` to the experiment output directory."""
    return write_manifest(manifest, output_dir / "manifest.json")
