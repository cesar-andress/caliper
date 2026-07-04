"""Tests for factorial experiment runner."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from caliper.config.loader import load_config
from caliper.runners.cells import expand_cells, make_cell_id
from caliper.runners.experiment import ExperimentRunner
from caliper.storage.formats import read_results


EXAMPLE_FACTORIAL = Path("configs/examples/example_factorial.yaml")
EXPECTED_CELLS = 2 * 2 * 2 * 2 * 2  # 32


class TestFactorialExpansion:
    def test_example_factorial_cell_count(self) -> None:
        config = load_config(EXAMPLE_FACTORIAL)
        assert config.total_combinations() == EXPECTED_CELLS
        assert len(expand_cells(config)) == EXPECTED_CELLS

    def test_expansion_is_deterministic(self) -> None:
        config = load_config(EXAMPLE_FACTORIAL)
        first = [make_cell_id(config, cell) for cell in expand_cells(config)]
        second = [make_cell_id(config, cell) for cell in expand_cells(config)]
        assert first == second

    def test_plan_matches_total_combinations(self) -> None:
        config = load_config(EXAMPLE_FACTORIAL)
        runner = ExperimentRunner(config, config_path=EXAMPLE_FACTORIAL, dry_run=True)
        assert len(runner.plan_combinations()) == EXPECTED_CELLS


class TestFactorialExecution:
    def test_full_run_produces_expected_rows(self, tmp_path: Path) -> None:
        config = load_config(EXAMPLE_FACTORIAL)
        config_dict = config.model_dump(mode="json")
        config_dict["output"] = {"directory": str(tmp_path / "outputs"), "format": "parquet"}
        config_dict["execution"] = {"shuffle": False, "parallel_workers": 1}

        config_path = tmp_path / "experiment.yaml"
        config_path.write_text(yaml.dump(config_dict), encoding="utf-8")

        loaded = load_config(config_path)
        runner = ExperimentRunner(loaded, config_path=config_path, dry_run=False)
        manifest = runner.run()

        assert manifest.status == "completed"
        assert manifest.total_cells == EXPECTED_CELLS
        assert manifest.completed_cells == EXPECTED_CELLS
        assert manifest.failed_cells == 0

        jsonl_path = runner.output_dir / "results.jsonl"
        parquet_path = runner.output_dir / "results.parquet"
        assert jsonl_path.exists()
        assert parquet_path.exists()

        df = read_results(parquet_path)
        assert len(df) == EXPECTED_CELLS
        assert set(df["status"]) == {"completed"}
        assert "cell_id" in df.columns
        assert "score" in df.columns
        assert df["experiment_id"].nunique() == 1

    def test_resume_skips_completed_cells(self, tmp_path: Path) -> None:
        config = load_config(EXAMPLE_FACTORIAL)
        config_dict = config.model_dump(mode="json")
        config_dict["output"] = {"directory": str(tmp_path / "outputs"), "format": "parquet"}
        config_dict["execution"] = {"shuffle": False, "parallel_workers": 1}
        config_dict["number_of_runs"] = 1
        config_dict["models"] = [config_dict["models"][0]]
        config_dict["tasks"] = [config_dict["tasks"][0]]
        config_dict["prompt_variants"] = [config_dict["prompt_variants"][0]]
        config_dict["temperatures"] = [0.0]

        config_path = tmp_path / "experiment.yaml"
        config_path.write_text(yaml.dump(config_dict), encoding="utf-8")
        loaded = load_config(config_path)
        expected_cells = 1

        runner = ExperimentRunner(loaded, config_path=config_path, dry_run=False)
        manifest = runner.run()
        assert manifest.completed_cells == expected_cells

        resume_runner = ExperimentRunner(
            loaded,
            config_path=config_path,
            dry_run=False,
            resume_dir=runner.output_dir,
        )
        resume_manifest = resume_runner.run()
        assert resume_manifest.skipped_cells == expected_cells
        assert resume_manifest.completed_cells == 0

        df = read_results(runner.output_dir / "results.parquet")
        assert len(df) == expected_cells

    def test_failed_cell_does_not_crash_experiment(self, tmp_path: Path, monkeypatch) -> None:
        from caliper.runners import executor as executor_module

        original_execute = executor_module.execute_cell

        def flaky_execute(**kwargs):
            cell = kwargs["cell"]
            if cell.run_index == 0:
                raise RuntimeError("simulated cell failure")
            return original_execute(**kwargs)

        monkeypatch.setattr(executor_module, "execute_cell", flaky_execute)

        config = load_config(EXAMPLE_FACTORIAL)
        config_dict = config.model_dump(mode="json")
        config_dict["output"] = {"directory": str(tmp_path / "outputs"), "format": "parquet"}
        config_dict["execution"] = {"shuffle": False, "parallel_workers": 1}
        config_dict["number_of_runs"] = 2
        config_dict["models"] = [config_dict["models"][0]]
        config_dict["tasks"] = [config_dict["tasks"][0]]
        config_dict["prompt_variants"] = [config_dict["prompt_variants"][0]]
        config_dict["temperatures"] = [0.0]

        config_path = tmp_path / "experiment.yaml"
        config_path.write_text(yaml.dump(config_dict), encoding="utf-8")
        loaded = load_config(config_path)

        runner = ExperimentRunner(loaded, config_path=config_path, dry_run=False)
        manifest = runner.run()

        assert manifest.status == "completed"
        assert manifest.failed_cells == 1
        assert manifest.completed_cells == 1

        df = pd.read_parquet(runner.output_dir / "results.parquet")
        assert len(df) == 2
        assert set(df["status"]) == {"completed", "failed"}

    def test_result_record_fields(self, tmp_path: Path) -> None:
        config = load_config(EXAMPLE_FACTORIAL)
        config_dict = config.model_dump(mode="json")
        config_dict["output"] = {"directory": str(tmp_path / "outputs"), "format": "parquet"}
        config_dict["execution"] = {"shuffle": False, "parallel_workers": 1}
        config_dict["number_of_runs"] = 1
        config_dict["models"] = [config_dict["models"][0]]
        config_dict["tasks"] = [config_dict["tasks"][0]]
        config_dict["prompt_variants"] = [config_dict["prompt_variants"][0]]
        config_dict["temperatures"] = [0.0]

        config_path = tmp_path / "experiment.yaml"
        config_path.write_text(yaml.dump(config_dict), encoding="utf-8")
        loaded = load_config(config_path)

        runner = ExperimentRunner(loaded, config_path=config_path, dry_run=False)
        runner.run()

        with (runner.output_dir / "results.jsonl").open(encoding="utf-8") as handle:
            record = json.loads(handle.readline())

        for field in (
            "cell_id",
            "experiment_id",
            "run_id",
            "run_index",
            "model_id",
            "provider_name",
            "provider_type",
            "task_id",
            "prompt_variant_id",
            "temperature",
            "metric",
            "score",
            "status",
            "executed_at",
            "metadata",
        ):
            assert field in record
