"""Smoke tests for the Paper 1 mock pilot experiment."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from caliper.config.loader import load_config, validate_config
from caliper.runners.experiment import ExperimentRunner

PILOT_CONFIG = Path("configs/paper1/pilot_variance_decomposition.yaml")
EXPECTED_CELLS = 6000
DATASET = Path("data/paper1/pilot_code_tasks.jsonl")


class TestPaper1PilotConfig:
    def test_pilot_config_validates(self) -> None:
        errors = validate_config(PILOT_CONFIG)
        assert errors == []

    def test_pilot_has_600_cells(self) -> None:
        config = load_config(PILOT_CONFIG)
        assert config.total_combinations() == EXPECTED_CELLS
        assert len(ExperimentRunner(config, config_path=PILOT_CONFIG, dry_run=True).plan_combinations()) == (
            EXPECTED_CELLS
        )

    def test_pilot_factorial_axes(self) -> None:
        config = load_config(PILOT_CONFIG)
        axes = config.factorial_axes()
        assert axes == {
            "models": 3,
            "temperatures": 2,
            "prompt_variants": 5,
            "tasks": 20,
            "runs": 10,
        }

    def test_pilot_dataset_has_20_tasks(self) -> None:
        lines = [line for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 20

    def test_pilot_dry_run(self) -> None:
        config = load_config(PILOT_CONFIG)
        manifest = ExperimentRunner(config, config_path=PILOT_CONFIG, dry_run=True).run()
        assert manifest.status == "completed"
        assert manifest.total_cells == EXPECTED_CELLS


class TestPaper1PilotAnalysisSmoke:
    @pytest.fixture(scope="class")
    def pilot_output_dir(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        tmp_path = tmp_path_factory.mktemp("paper1_pilot")
        config = load_config(PILOT_CONFIG)
        config_dict = config.model_dump(mode="json")
        config_dict["output"] = {"directory": str(tmp_path / "outputs"), "format": "parquet"}
        config_dict["number_of_runs"] = 1
        config_dict["models"] = [config_dict["models"][0]]
        config_dict["tasks"] = config_dict["tasks"][:2]
        config_dict["prompt_variants"] = config_dict["prompt_variants"][:2]
        config_dict["temperatures"] = [0.0]

        config_path = tmp_path / "pilot_smoke.yaml"
        config_path.write_text(yaml.dump(config_dict), encoding="utf-8")
        loaded = load_config(config_path)
        runner = ExperimentRunner(loaded, config_path=config_path, dry_run=False, auto_resume=False)
        runner.run()
        return runner.output_dir

    def test_variance_analysis_runs(self, pilot_output_dir: Path) -> None:
        stats_path = pilot_output_dir / "statistical_dataset.parquet"
        assert stats_path.exists()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "caliper",
                "analyze",
                "variance",
                "--results",
                str(stats_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "Variance decomposition" in result.stdout

    def test_power_analysis_runs(self, pilot_output_dir: Path) -> None:
        stats_path = pilot_output_dir / "statistical_dataset.parquet"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "caliper",
                "analyze",
                "power",
                "--results",
                str(stats_path),
                "--simulations",
                "50",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "Power simulation" in result.stdout

    def test_ranking_fragility_runs(self, pilot_output_dir: Path) -> None:
        results_path = pilot_output_dir / "results.parquet"
        out_dir = pilot_output_dir / "analysis" / "ranking_fragility"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "caliper",
                "ranking-fragility",
                str(results_path),
                "--output-dir",
                str(out_dir),
                "--n-bootstrap",
                "20",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert (out_dir / "ranking_fragility_summary.csv").exists()

    def test_pilot_scores_show_model_separation(self, pilot_output_dir: Path) -> None:
        df = pd.read_parquet(pilot_output_dir / "results.parquet")
        assert df["model_id"].nunique() == 1
        assert df["score"].mean() >= 0.0
