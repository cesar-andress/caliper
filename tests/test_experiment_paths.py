"""Tests for completed experiment directory resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from caliper.runners.experiment_paths import (
    ExperimentDirectoryError,
    resolve_experiment_dir,
    validate_experiment_data_files,
)


def _write_completed_experiment(path: Path, *, with_stats: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text('{"experiment_id": "test"}', encoding="utf-8")
    (path / "results.parquet").write_text("placeholder", encoding="utf-8")
    if with_stats:
        (path / "statistical_dataset.parquet").write_text("placeholder", encoding="utf-8")


class TestResolveExperimentDir:
    def test_direct_experiment_directory(self, tmp_path: Path) -> None:
        exp = tmp_path / "run_a"
        _write_completed_experiment(exp)
        assert resolve_experiment_dir(exp) == exp.resolve()

    def test_parent_directory_with_one_nested_experiment(self, tmp_path: Path) -> None:
        parent = tmp_path / "paper1_confirmatory_humaneval"
        exp = parent / "paper1_confirmatory_humaneval"
        _write_completed_experiment(exp)
        assert resolve_experiment_dir(parent) == exp.resolve()

    def test_missing_experiment_path(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"
        with pytest.raises(ExperimentDirectoryError, match="does not exist"):
            resolve_experiment_dir(missing)

    def test_parent_with_multiple_candidate_experiments(self, tmp_path: Path) -> None:
        parent = tmp_path / "experiments"
        _write_completed_experiment(parent / "run_a")
        _write_completed_experiment(parent / "run_b")
        with pytest.raises(ExperimentDirectoryError, match="Multiple completed"):
            resolve_experiment_dir(parent)

    def test_directory_missing_manifest_json(self, tmp_path: Path) -> None:
        path = tmp_path / "empty_parent"
        path.mkdir()
        with pytest.raises(ExperimentDirectoryError, match="No completed experiment"):
            resolve_experiment_dir(path)

    def test_directory_missing_results_parquet(self, tmp_path: Path) -> None:
        path = tmp_path / "partial"
        path.mkdir()
        (path / "manifest.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ExperimentDirectoryError, match="missing required files"):
            resolve_experiment_dir(path)

    def test_directory_missing_results_and_statistical_dataset(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest_only"
        path.mkdir()
        (path / "manifest.json").write_text("{}", encoding="utf-8")
        (path / "results.parquet").write_text("placeholder", encoding="utf-8")
        (path / "results.parquet").unlink()
        (path / "manifest.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ExperimentDirectoryError, match="missing required files"):
            resolve_experiment_dir(path)

    def test_validate_experiment_data_files_requires_analysis_input(self, tmp_path: Path) -> None:
        path = tmp_path / "no_data"
        path.mkdir()
        (path / "manifest.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ExperimentDirectoryError, match="missing analysis data"):
            validate_experiment_data_files(path)

    def test_error_message_includes_requested_checked_expected_candidates(
        self,
        tmp_path: Path,
    ) -> None:
        parent = tmp_path / "parent"
        _write_completed_experiment(parent / "run_a")
        _write_completed_experiment(parent / "run_b")
        with pytest.raises(ExperimentDirectoryError) as exc_info:
            resolve_experiment_dir(parent)
        message = str(exc_info.value)
        assert "Requested path:" in message
        assert "Paths checked:" in message
        assert "Expected files:" in message
        assert "Valid candidate directories:" in message
        assert str(parent / "run_a") in message
        assert str(parent / "run_b") in message
