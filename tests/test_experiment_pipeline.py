"""Tests for reproducible experiment pipeline infrastructure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from caliper.config.loader import load_config
from caliper.runners.cells import expand_cells, make_cell_id
from caliper.runners.checkpoint import CheckpointStore
from caliper.runners.experiment import (
    EXPERIMENTS_ROOT,
    ExperimentRunner,
    detect_resume_dir,
    resolve_experiment_output_dir,
)
from caliper.runners.progress import ExecutionProgress
from caliper.runners.reproducibility import (
    cell_seed,
    configuration_hash,
    experiment_hash,
    make_cell_hash,
)
from caliper.runners.results import ExperimentResultRecord
from caliper.storage.formats import read_results

EXAMPLE_FACTORIAL = Path("configs/examples/example_factorial.yaml")


class TestDeterministicHashing:
    def test_cell_hash_includes_seed(self) -> None:
        config = load_config(EXAMPLE_FACTORIAL)
        cell = next(iter(expand_cells(config)))
        seed = cell_seed(config, cell)
        h1 = make_cell_hash(config, cell)
        h2 = make_cell_hash(config, cell)
        assert h1 == h2
        assert len(h1) == 64
        assert make_cell_id(config, cell) == h1

        config_other_seed = config.model_copy(update={"random_seed": config.random_seed + 1})
        assert make_cell_hash(config_other_seed, cell) != h1
        assert cell_seed(config, cell) == seed

    def test_configuration_and_experiment_hashes_are_stable(self) -> None:
        config = load_config(EXAMPLE_FACTORIAL)
        cells = expand_cells(config)
        cell_hashes = [make_cell_hash(config, cell) for cell in cells]
        assert configuration_hash(config) == configuration_hash(config)
        assert experiment_hash(config, cell_hashes) == experiment_hash(config, cell_hashes)


class TestExecutionProgress:
    def test_progress_tracks_eta_and_throughput(self) -> None:
        progress = ExecutionProgress(total_cells=4)
        progress.record_completion(success=True)
        progress.record_completion(success=False)
        snapshot = progress.to_dict()
        assert snapshot["completed_cells"] == 1
        assert snapshot["failed_cells"] == 1
        assert snapshot["pending_cells"] == 2
        assert snapshot["throughput_cells_per_second"] >= 0


class TestCheckpointStore:
    def test_checkpoint_persists_completed_cell(self, tmp_path: Path) -> None:
        store = CheckpointStore(tmp_path / "checkpoints")
        record = ExperimentResultRecord(
            cell_id="abc123",
            experiment_id="demo",
            run_id="run1",
            run_index=0,
            model_id="m1",
            provider_name="mock",
            provider_type="mock",
            task_id="t1",
            prompt_variant_id="p1",
            temperature=0.0,
            metric="accuracy",
            score=1.0,
            status="completed",
        )
        store.write(record)
        assert store.load_completed_cell_ids() == {"abc123"}


class TestOutputLayout:
    def test_default_output_uses_experiments_root(self) -> None:
        config = load_config(EXAMPLE_FACTORIAL)
        out = resolve_experiment_output_dir(config)
        assert out == EXPERIMENTS_ROOT / config.experiment_id

    def test_custom_output_directory_is_respected(self, tmp_path: Path) -> None:
        config = load_config(EXAMPLE_FACTORIAL)
        config = config.model_copy(
            update={"output": config.output.model_copy(update={"directory": tmp_path / "custom"})}
        )
        out = resolve_experiment_output_dir(config)
        assert out == tmp_path / "custom" / config.experiment_id


class TestPipelineArtifacts:
    def _tiny_config(self, tmp_path: Path):
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
        return load_config(config_path), config_path

    def test_full_pipeline_writes_required_artifacts(self, tmp_path: Path) -> None:
        loaded, config_path = self._tiny_config(tmp_path)
        runner = ExperimentRunner(
            loaded,
            config_path=config_path,
            dry_run=False,
            auto_resume=False,
        )
        manifest = runner.run()

        out = runner.output_dir
        assert manifest.status == "completed"
        for name in (
            "manifest.json",
            "config.yaml",
            "results.parquet",
            "results.jsonl",
            "report.md",
        ):
            assert (out / name).exists(), name

        assert (out / "logs").is_dir()
        assert (out / "figures").is_dir()
        assert (out / "checkpoints").is_dir()
        assert (out / "statistical_dataset.parquet").exists()

        manifest_data = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        for key in (
            "software_version",
            "git_commit",
            "python_version",
            "os",
            "cpu",
            "gpu",
            "libraries",
            "random_seed",
            "experiment_hash",
            "configuration_hash",
            "execution_duration_seconds",
            "completed_cells",
            "failed_cells",
        ):
            assert key in manifest_data, key

        report = (out / "report.md").read_text(encoding="utf-8")
        assert "# Experiment report:" in report
        assert "## Reproducibility" in report

    def test_auto_resume_skips_completed_experiment(self, tmp_path: Path) -> None:
        loaded, config_path = self._tiny_config(tmp_path)
        runner = ExperimentRunner(
            loaded,
            config_path=config_path,
            dry_run=False,
            auto_resume=False,
        )
        runner.run()
        assert detect_resume_dir(loaded) is None

    def test_auto_resume_continues_partial_run(self, tmp_path: Path) -> None:
        loaded, config_path = self._tiny_config(tmp_path)
        config_dict = loaded.model_dump(mode="json")
        config_dict["number_of_runs"] = 2
        config_path.write_text(yaml.dump(config_dict), encoding="utf-8")
        loaded = load_config(config_path)

        out = resolve_experiment_output_dir(loaded)
        out.mkdir(parents=True)
        partial_record = {
            "cell_id": make_cell_id(loaded, next(iter(expand_cells(loaded)))),
            "experiment_id": loaded.experiment_id,
            "run_id": "partial",
            "run_index": 0,
            "model_id": loaded.models[0].id,
            "provider_name": "mock",
            "provider_type": "mock",
            "task_id": loaded.tasks[0].id,
            "prompt_variant_id": loaded.prompt_variants[0].id,
            "temperature": 0.0,
            "metric": "accuracy",
            "score": 1.0,
            "status": "completed",
        }
        (out / "results.jsonl").write_text(json.dumps(partial_record) + "\n", encoding="utf-8")

        assert detect_resume_dir(loaded) == out

        resume_runner = ExperimentRunner(
            loaded,
            config_path=config_path,
            dry_run=False,
            auto_resume=True,
        )
        resume_manifest = resume_runner.run()
        assert resume_manifest.skipped_cells >= 1
        assert resume_manifest.completed_cells >= 1

        df = read_results(out / "results.parquet")
        assert len(df) == 2

    def test_checkpoint_enables_resume_after_interruption(self, tmp_path: Path, monkeypatch) -> None:
        from caliper.runners import results as results_module

        original_append = results_module.ResultWriter.append
        calls = {"count": 0}

        def append_and_interrupt(self, record):
            calls["count"] += 1
            path = original_append(self, record)
            if calls["count"] == 1:
                raise RuntimeError("simulated interruption")
            return path

        monkeypatch.setattr(results_module.ResultWriter, "append", append_and_interrupt)

        loaded, config_path = self._tiny_config(tmp_path)
        config_dict = loaded.model_dump(mode="json")
        config_dict["number_of_runs"] = 2
        config_path.write_text(yaml.dump(config_dict), encoding="utf-8")
        loaded = load_config(config_path)

        runner = ExperimentRunner(
            loaded,
            config_path=config_path,
            dry_run=False,
            auto_resume=False,
        )
        with pytest.raises(RuntimeError, match="simulated interruption"):
            runner.run()

        resume_runner = ExperimentRunner(
            loaded,
            config_path=config_path,
            dry_run=False,
            resume_dir=runner.output_dir,
        )
        resume_manifest = resume_runner.run()
        assert resume_manifest.skipped_cells >= 1
        assert resume_manifest.completed_cells >= 1

        checkpoints = list((runner.output_dir / "checkpoints").glob("*.json"))
        assert checkpoints
