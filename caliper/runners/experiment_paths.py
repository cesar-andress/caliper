"""Resolve completed experiment directories from flat or nested layouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REQUIRED_RESOLUTION_FILES = ("manifest.json", "results.parquet")
ANALYSIS_DATA_FILES = ("results.parquet", "statistical_dataset.parquet")


@dataclass
class ExperimentDirectoryError(Exception):
    """Raised when an experiment directory cannot be resolved unambiguously."""

    message: str
    requested_path: Path
    paths_checked: list[Path]
    expected_files: tuple[str, ...]
    candidates: list[Path]

    def __str__(self) -> str:
        checked = "\n".join(f"  - {path}" for path in self.paths_checked) or "  - (none)"
        expected = ", ".join(self.expected_files)
        candidate_lines = "\n".join(f"  - {path}" for path in self.candidates) or "  - (none)"
        return (
            f"{self.message}\n"
            f"Requested path: {self.requested_path}\n"
            f"Paths checked:\n{checked}\n"
            f"Expected files: {expected}\n"
            f"Valid candidate directories:\n{candidate_lines}"
        )


def _is_resolution_candidate(path: Path) -> bool:
    return all((path / name).is_file() for name in REQUIRED_RESOLUTION_FILES)


def validate_experiment_data_files(experiment_dir: Path) -> None:
    """Ensure a resolved experiment directory contains analysis inputs."""
    experiment_dir = experiment_dir.resolve()
    if not (experiment_dir / "manifest.json").is_file():
        raise ExperimentDirectoryError(
            f"Resolved experiment directory is missing manifest.json: {experiment_dir}",
            requested_path=experiment_dir,
            paths_checked=[experiment_dir],
            expected_files=("manifest.json", *ANALYSIS_DATA_FILES),
            candidates=[],
        )

    if not any((experiment_dir / name).is_file() for name in ANALYSIS_DATA_FILES):
        raise ExperimentDirectoryError(
            "Resolved experiment directory is missing analysis data files "
            f"({', '.join(ANALYSIS_DATA_FILES)}).",
            requested_path=experiment_dir,
            paths_checked=[experiment_dir],
            expected_files=ANALYSIS_DATA_FILES,
            candidates=[],
        )


def resolve_experiment_dir(requested_path: Path | str) -> Path:
    """Resolve a completed experiment directory from a direct or parent path."""
    requested = Path(requested_path).expanduser()
    if not requested.exists():
        raise ExperimentDirectoryError(
            f"Experiment path does not exist: {requested}",
            requested_path=requested,
            paths_checked=[],
            expected_files=REQUIRED_RESOLUTION_FILES,
            candidates=[],
        )

    requested = requested.resolve()
    paths_checked = [requested]

    if (requested / "manifest.json").is_file():
        if _is_resolution_candidate(requested):
            validate_experiment_data_files(requested)
            return requested
        missing = [
            name for name in REQUIRED_RESOLUTION_FILES if not (requested / name).is_file()
        ]
        raise ExperimentDirectoryError(
            f"Directory contains manifest.json but is missing required files: {', '.join(missing)}",
            requested_path=requested,
            paths_checked=paths_checked,
            expected_files=REQUIRED_RESOLUTION_FILES,
            candidates=[],
        )

    if not requested.is_dir():
        raise ExperimentDirectoryError(
            f"Experiment path is not a directory and has no manifest.json: {requested}",
            requested_path=requested,
            paths_checked=paths_checked,
            expected_files=REQUIRED_RESOLUTION_FILES,
            candidates=[],
        )

    candidates = [
        child.resolve()
        for child in sorted(requested.iterdir())
        if child.is_dir() and _is_resolution_candidate(child)
    ]
    paths_checked.extend(candidates)

    if len(candidates) == 1:
        validate_experiment_data_files(candidates[0])
        return candidates[0]

    if not candidates:
        raise ExperimentDirectoryError(
            "No completed experiment directory found. "
            "Expected manifest.json and results.parquet in the requested path "
            "or in exactly one immediate child directory.",
            requested_path=requested,
            paths_checked=paths_checked,
            expected_files=REQUIRED_RESOLUTION_FILES,
            candidates=[],
        )

    raise ExperimentDirectoryError(
        "Multiple completed experiment directories found under the requested path.",
        requested_path=requested,
        paths_checked=paths_checked,
        expected_files=REQUIRED_RESOLUTION_FILES,
        candidates=candidates,
    )
