"""Tests for missing-cell diagnostics, failure traceability, and recovery."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from click.testing import CliRunner

from caliper.cli import main
from caliper.config.loader import load_config
from caliper.models.errors import ProviderGenerationError
from caliper.models.retry import RetryPolicy, execute_with_retry_and_timeout
from caliper.runners.cells import expand_cells, make_cell_id
from caliper.runners.checkpoint import CheckpointStore
from caliper.runners.experiment import ExperimentRunner
from caliper.runners.failures import (
    count_terminal_failures,
    duplicate_cell_ids,
    load_results_records,
)
from caliper.runners.missing_cells import inspect_missing_cells, write_missing_cells_report
from caliper.runners.results import ExperimentResultRecord
from caliper.runners.retry_missing import resolve_config_path, retry_missing_cells
from caliper.storage.formats import read_results

EXAMPLE_FACTORIAL = Path("configs/examples/example_factorial.yaml")


def _minimal_config_dict(tmp_path: Path) -> dict:
    config = load_config(EXAMPLE_FACTORIAL)
    config_dict = config.model_dump(mode="json")
    config_dict["output"] = {"directory": str(tmp_path / "outputs"), "format": "parquet"}
    config_dict["execution"] = {"shuffle": False, "parallel_workers": 1}
    config_dict["number_of_runs"] = 2
    config_dict["models"] = [config_dict["models"][0]]
    config_dict["tasks"] = [config_dict["tasks"][0]]
    config_dict["prompt_variants"] = [config_dict["prompt_variants"][0]]
    config_dict["temperatures"] = [0.0]
    return config_dict


def _write_config(tmp_path: Path, config_dict: dict) -> Path:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.dump(config_dict), encoding="utf-8")
    return config_path


class TestMissingCellInspection:
    def test_detects_two_missing_cells(self, tmp_path: Path) -> None:
        config_dict = _minimal_config_dict(tmp_path)
        config_path = _write_config(tmp_path, config_dict)
        loaded = load_config(config_path)

        runner = ExperimentRunner(loaded, config_path=config_path, dry_run=False)
        manifest = runner.run()
        assert manifest.completed_cells == 2

        cells = expand_cells(loaded)
        missing_specs = cells[:2]
        for cell in missing_specs:
            (runner.output_dir / "checkpoints" / f"{make_cell_id(loaded, cell)}.json").unlink()

        report = inspect_missing_cells(
            runner.output_dir,
            loaded,
            config_dir=config_path.parent.resolve(),
        )
        assert report["counts"]["expected_cells"] == 2
        assert report["counts"]["missing_cell_ids"] == 2
        assert len(report["missing_cells"]) == 2
        for missing in report["missing_cells"]:
            assert "rendered_prompt_hash" in missing

    def test_detects_duplicate_cells(self, tmp_path: Path) -> None:
        config_dict = _minimal_config_dict(tmp_path)
        config_dict["number_of_runs"] = 1
        config_path = _write_config(tmp_path, config_dict)
        loaded = load_config(config_path)

        runner = ExperimentRunner(loaded, config_path=config_path, dry_run=False)
        runner.run()

        cell = expand_cells(loaded)[0]
        cell_id = make_cell_id(loaded, cell)
        duplicate = ExperimentResultRecord(
            cell_id=cell_id,
            experiment_id=loaded.experiment_id,
            run_id="dup-run",
            run_index=cell.run_index,
            model_id=cell.model_id,
            provider_name="mock",
            provider_type="mock",
            task_id=cell.task_id,
            prompt_variant_id=cell.prompt_variant_id,
            temperature=cell.temperature,
            seed=loaded.random_seed + cell.run_index,
            metric="exact_match",
            score=1.0,
            status="completed",
        )
        with (runner.output_dir / "results.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(duplicate.model_dump_json())
            handle.write("\n")

        records = load_results_records(runner.output_dir / "results.jsonl")
        assert duplicate_cell_ids(records) == [cell_id]

        report = inspect_missing_cells(
            runner.output_dir,
            loaded,
            config_dir=config_path.parent.resolve(),
        )
        assert report["counts"]["duplicate_cell_ids"] == 1
        assert report["duplicate_cell_ids"] == [cell_id]


class TestFailureTraceability:
    def test_terminal_failure_writes_failures_jsonl_and_failed_checkpoint(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        from caliper.runners import executor as executor_module

        original_execute = executor_module.execute_cell

        def flaky_execute(**kwargs):
            cell = kwargs["cell"]
            if cell.run_index == 0:
                raise RuntimeError("terminal provider failure")
            return original_execute(**kwargs)

        monkeypatch.setattr(executor_module, "execute_cell", flaky_execute)

        config_dict = _minimal_config_dict(tmp_path)
        config_path = _write_config(tmp_path, config_dict)
        loaded = load_config(config_path)

        runner = ExperimentRunner(loaded, config_path=config_path, dry_run=False)
        manifest = runner.run()

        assert manifest.failed_cells == 1
        assert (runner.output_dir / "failures.jsonl").exists()
        failed_records = load_results_records(runner.output_dir / "failures.jsonl")
        assert len(failed_records) == 1
        assert failed_records[0].status == "failed"

        checkpoint_store = CheckpointStore(runner.output_dir / "checkpoints")
        failed_ids = checkpoint_store.load_failed_cell_ids()
        assert len(failed_ids) == 1

    def test_transient_provider_retries_do_not_count_as_terminal_failures(self) -> None:
        attempts = {"count": 0}

        def operation() -> str:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise ProviderGenerationError("transient", provider_name="mock", retryable=True)
            return "ok"

        result = execute_with_retry_and_timeout(
            operation,
            provider_name="mock",
            timeout_seconds=5.0,
            retry=RetryPolicy(max_retries=3),
        )
        assert result == "ok"
        assert attempts["count"] == 3


class TestRetryMissingCells:
    def test_resolve_config_path_from_manifest_when_report_lacks_config(
        self,
        tmp_path: Path,
    ) -> None:
        config_dict = _minimal_config_dict(tmp_path)
        config_path = _write_config(tmp_path, config_dict)
        loaded = load_config(config_path)

        runner = ExperimentRunner(loaded, config_path=config_path, dry_run=False)
        runner.run()

        report = inspect_missing_cells(
            runner.output_dir,
            loaded,
            config_dir=config_path.parent.resolve(),
        )
        report_path = write_missing_cells_report(runner.output_dir, report)["json"]

        resolved = resolve_config_path(
            runner.output_dir,
            report,
            report_path=report_path,
        )
        assert resolved.resolve() == config_path.resolve()

    def test_retry_missing_cells_recovers_without_overwriting_completed(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        from caliper.runners import executor as executor_module

        original_execute = executor_module.execute_cell
        fail_once = {"done": False}

        def flaky_execute(**kwargs):
            cell = kwargs["cell"]
            if cell.run_index == 0 and not fail_once["done"]:
                fail_once["done"] = True
                raise RuntimeError("missing cell failure")
            return original_execute(**kwargs)

        monkeypatch.setattr(executor_module, "execute_cell", flaky_execute)

        config_dict = _minimal_config_dict(tmp_path)
        config_path = _write_config(tmp_path, config_dict)
        loaded = load_config(config_path)

        runner = ExperimentRunner(loaded, config_path=config_path, dry_run=False)
        runner.run()

        report = inspect_missing_cells(
            runner.output_dir,
            loaded,
            config_dir=config_path.parent.resolve(),
        )
        assert report["counts"]["missing_cell_ids"] == 1
        report_path = write_missing_cells_report(
            runner.output_dir,
            report,
            write_retry_config=True,
            config_path=config_path,
        )["json"]

        completed_before = {
            json.loads(line)["cell_id"]
            for line in (runner.output_dir / "results.jsonl").read_text().splitlines()
            if json.loads(line)["status"] == "completed"
        }

        summary = retry_missing_cells(
            runner.output_dir,
            report_path=report_path,
            config_path=config_path,
        )
        assert summary["recovered_cells"] == 1
        assert summary["remaining_missing_cells"] == 0
        assert (runner.output_dir / "recovery_audit.jsonl").exists()

        df = read_results(runner.output_dir / "results.parquet")
        latest_completed = {
            row["cell_id"]
            for _, row in df.sort_values("executed_at").groupby("cell_id").tail(1).iterrows()
            if row["status"] == "completed"
        }
        assert completed_before.issubset(latest_completed)
        assert count_terminal_failures(runner.output_dir / "results.jsonl") == 0


class TestInspectMissingCellsCLI:
    def test_cli_writes_reports(self, tmp_path: Path) -> None:
        config_dict = _minimal_config_dict(tmp_path)
        config_dict["number_of_runs"] = 1
        config_path = _write_config(tmp_path, config_dict)
        loaded = load_config(config_path)

        runner = ExperimentRunner(loaded, config_path=config_path, dry_run=False)
        runner.run()

        cell = expand_cells(loaded)[0]
        checkpoint_path = runner.output_dir / "checkpoints" / f"{make_cell_id(loaded, cell)}.json"
        checkpoint_path.unlink()

        runner_cli = CliRunner()
        result = runner_cli.invoke(
            main,
            [
                "inspect-missing-cells",
                str(runner.output_dir),
                "--config",
                str(config_path),
                "--write-retry-config",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (runner.output_dir / "missing_cells_report.json").exists()
        assert (runner.output_dir / "missing_cells_report.md").exists()
        assert (runner.output_dir / "retry_missing_cells.json").exists()
