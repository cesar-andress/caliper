"""Deterministic hashes and environment metadata for reproducible experiments."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from typing import Any

from caliper import __version__
from caliper.config.schema import ExperimentCombination, ExperimentConfig

LIBRARY_PACKAGES = (
    "caliper",
    "pandas",
    "numpy",
    "pydantic",
    "structlog",
    "click",
    "pyarrow",
    "scipy",
    "statsmodels",
)


def cell_seed(config: ExperimentConfig, cell: ExperimentCombination) -> int:
    """Per-cell RNG seed derived deterministically from the experiment seed."""
    return config.random_seed + cell.run_index


def make_cell_hash(config: ExperimentConfig, cell: ExperimentCombination) -> str:
    """Return a stable SHA-256 hash identifying one factorial cell forever.

    Hash inputs: model, task, prompt, temperature, run index, seed.
    """
    seed = cell_seed(config, cell)
    key = "|".join(
        [
            cell.model_id,
            cell.task_id,
            cell.prompt_variant_id,
            f"{cell.temperature:.6f}",
            str(cell.run_index),
            str(seed),
        ]
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def configuration_hash(config: ExperimentConfig) -> str:
    """Hash the canonical experiment configuration."""
    payload = config.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def experiment_hash(config: ExperimentConfig, cell_hashes: list[str]) -> str:
    """Hash the full experiment design including every cell identity."""
    design = "|".join([config.experiment_id, configuration_hash(config), *sorted(cell_hashes)])
    return hashlib.sha256(design.encode("utf-8")).hexdigest()


def git_commit() -> str:
    """Return the current git commit hash, or ``unknown`` if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def collect_cpu_info() -> dict[str, Any]:
    """Collect CPU metadata."""
    return {
        "processor": platform.processor() or "unknown",
        "physical_cores": platform.processor(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }


def collect_gpu_info() -> dict[str, Any]:
    """Collect GPU metadata when CUDA/NVML is available."""
    try:
        from caliper.models.local.metadata import collect_gpu_metadata

        return collect_gpu_metadata(device="cuda").to_dict()
    except Exception:
        return {"gpu_available": False, "device": "cpu"}


def collect_library_versions() -> dict[str, str]:
    """Collect versions of key runtime libraries."""
    versions: dict[str, str] = {"python": platform.python_version()}
    for package in LIBRARY_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            if package == "caliper":
                versions[package] = __version__
            else:
                versions[package] = "not_installed"
    return versions


def collect_environment() -> dict[str, Any]:
    """Collect full reproducibility environment metadata."""
    return {
        "software_version": __version__,
        "git_commit": git_commit(),
        "python_version": sys.version,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "cpu": collect_cpu_info(),
        "gpu": collect_gpu_info(),
        "libraries": collect_library_versions(),
    }
