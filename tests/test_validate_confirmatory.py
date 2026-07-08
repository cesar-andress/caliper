"""Regression tests for confirmatory pre-flight validation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from caliper.cli import main
from caliper.validation.confirmatory import (
    run_confirmatory_validation,
    validate_benchmark_load,
    validate_model_provider,
    validate_ollama_connectivity,
    validate_task_metadata,
    validate_timeout_handling,
)
from caliper.validation.config_builder import build_preflight_config
from caliper.validation.types import ValidationStage


FIXTURE_DATASET = Path("tests/fixtures/benchmarks/sample_tasks.jsonl")


class TestValidateBenchmarkLoad:
    def test_fails_when_benchmark_missing(self, tmp_path: Path) -> None:
        with patch("caliper.validation.confirmatory.dataset_path", return_value=tmp_path / "missing.jsonl"):
            stage, _ = validate_benchmark_load("humaneval", num_tasks=3)
        assert stage.status == "FAIL"
        assert stage.stage == ValidationStage.BENCHMARK_LOAD
        assert stage.root_cause == "Benchmark JSONL not materialized"

    def test_passes_with_fixture_dataset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "caliper.validation.confirmatory.dataset_path",
            lambda _b: FIXTURE_DATASET,
        )
        stage, info = validate_benchmark_load("humaneval", num_tasks=2)
        assert stage.status == "PASS"
        assert info["num_tasks"] >= 2


class TestValidateTaskMetadata:
    def test_fails_on_missing_task_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "caliper.validation.confirmatory.dataset_path",
            lambda _b: FIXTURE_DATASET,
        )
        stage = validate_task_metadata("humaneval", ["nonexistent-task"])
        assert stage.status == "FAIL"
        assert "Missing task ids" in stage.message


class TestValidateTimeout:
    def test_timeout_handling_passes(self) -> None:
        stage = validate_timeout_handling()
        assert stage.status == "PASS"
        assert stage.stage == ValidationStage.TIMEOUT_HANDLING


class TestPreflightConfig:
    def test_builds_minimal_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "caliper.validation.config_builder.dataset_path",
            lambda _b: FIXTURE_DATASET,
        )
        monkeypatch.setattr(
            "caliper.validation.config_builder.REFERENCE_CONFIGS",
            {
                "humaneval_plus": Path("configs/paper1/confirmatory_humaneval.yaml"),
                "mbpp": Path("configs/paper1/confirmatory_mbpp.yaml"),
            },
        )
        monkeypatch.setattr(
            "caliper.validation.config_builder.DATASET_PATHS",
            {"humaneval_plus": FIXTURE_DATASET, "mbpp": FIXTURE_DATASET},
        )
        config, ref_path, task_ids = build_preflight_config(
            "humaneval",
            num_tasks=2,
            model_id="qwen25_coder_7b",
            prompt_id="minimal",
        )
        assert config.experiment_id.startswith("preflight_validate_")
        assert len(config.tasks) == 2
        assert len(config.models) == 1
        assert len(task_ids) == 2
        assert ref_path.exists()


class TestValidateConfirmatoryCLI:
    def setup_method(self) -> None:
        self.runner = CliRunner()

    def test_cli_help(self) -> None:
        result = self.runner.invoke(main, ["validate-confirmatory", "--help"])
        assert result.exit_code == 0
        assert "--benchmark" in result.output
        assert "--tasks" in result.output

    @patch("caliper.validation.confirmatory.run_confirmatory_validation")
    def test_cli_success_exit_code(self, mock_run: MagicMock) -> None:
        from caliper.validation.types import TimingBreakdown, ValidationReport

        mock_run.return_value = ValidationReport(
            benchmark="humaneval_plus",
            output_dir="/tmp/preflight",
            stages=[],
            timing=TimingBreakdown(observations=3, pipeline_latency_ms=3000),
            environment={},
            benchmark_info={},
            sanity={},
            warnings=[],
            ready_to_launch=True,
        )
        result = self.runner.invoke(main, ["validate-confirmatory", "--benchmark", "humaneval"])
        assert result.exit_code == 0
        assert "Ready to launch: YES" in result.output

    @patch("caliper.validation.confirmatory.run_confirmatory_validation")
    def test_cli_failure_exit_code(self, mock_run: MagicMock) -> None:
        from caliper.validation.types import StageResult, TimingBreakdown, ValidationReport

        mock_run.return_value = ValidationReport(
            benchmark="humaneval_plus",
            output_dir="/tmp/preflight",
            stages=[
                StageResult(
                    stage=ValidationStage.OLLAMA_CONNECTIVITY,
                    status="FAIL",
                    message="down",
                    root_cause="Ollama unavailable",
                    recommended_fix="Start Ollama",
                    severity="critical",
                )
            ],
            timing=TimingBreakdown(),
            environment={},
            benchmark_info={},
            sanity={},
            warnings=[],
            ready_to_launch=False,
        )
        result = self.runner.invoke(main, ["validate-confirmatory", "--benchmark", "humaneval"])
        assert result.exit_code == 1
        assert "Ready to launch: NO" in result.output


class TestProviderFailures:
    def test_model_provider_fails_when_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "caliper.validation.config_builder.dataset_path",
            lambda _b: FIXTURE_DATASET,
        )
        config, _, _ = build_preflight_config("humaneval", num_tasks=1)
        with patch("caliper.validation.confirmatory.build_provider") as mock_build:
            mock_provider = MagicMock()
            mock_provider.is_available.return_value = False
            mock_build.return_value = mock_provider
            stage = validate_model_provider(config, "qwen25_coder_7b")
        assert stage.status == "FAIL"
        assert stage.root_cause == "Model provider unavailable"

    def test_ollama_fails_when_model_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "caliper.validation.config_builder.dataset_path",
            lambda _b: FIXTURE_DATASET,
        )
        config, _, _ = build_preflight_config("humaneval", num_tasks=1)
        with patch("caliper.models.ollama_provider.list_local_models", return_value=["other:model"]):
            stage = validate_ollama_connectivity(config, "qwen25_coder_7b")
        assert stage.status == "FAIL"
        assert stage.root_cause == "Model missing from Ollama"


class TestRunConfirmatoryValidationDry:
    def test_fails_fast_when_benchmark_missing(self, tmp_path: Path) -> None:
        with patch("caliper.validation.confirmatory.dataset_path", return_value=tmp_path / "nope.jsonl"):
            report = run_confirmatory_validation(benchmark="humaneval", skip_experiment=True)
        assert report.ready_to_launch is False
        assert any(s.stage == ValidationStage.BENCHMARK_LOAD and s.failed for s in report.stages)

    def test_invalid_configuration_reference(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "caliper.validation.confirmatory.validate_config",
            lambda _p: ["invalid config"],
        )
        monkeypatch.setattr(
            "caliper.validation.confirmatory.dataset_path",
            lambda _b: FIXTURE_DATASET,
        )
        report = run_confirmatory_validation(benchmark="humaneval", skip_experiment=True)
        assert not report.ready_to_launch


@pytest.mark.integration
class TestValidateConfirmatoryLive:
    def test_end_to_end_preflight_humaneval(self) -> None:
        dataset = Path("data/benchmarks/humaneval_plus.jsonl")
        if not dataset.exists():
            pytest.skip("Benchmark not materialized")
        report = run_confirmatory_validation(
            benchmark="humaneval",
            tasks=1,
            runs=1,
            verbose=False,
        )
        assert report.output_dir
        assert Path(report.output_dir, "validation_report.md").exists()
        assert Path(report.output_dir, "launch_checklist.md").exists()
