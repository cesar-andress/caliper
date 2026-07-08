"""Tests for primary metric configuration and report selection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from caliper.config.loader import load_config
from caliper.config.metrics import resolve_analysis_metric, resolve_primary_metric
from caliper.config.schema import ExperimentConfig
from caliper.runners.report import generate_report
from caliper.statistics.prepare import prepare_results_table


@pytest.fixture
def code_results_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_id": "model-a",
                "task_id": "task-1",
                "prompt_variant_id": "direct",
                "run_index": 0,
                "temperature": 0.0,
                "status": "completed",
                "metric": "exact_match",
                "score": 0.0,
                "scores": {
                    "exact_match": 0.0,
                    "normalized_code_match": 1.0,
                    "syntax_check": 1.0,
                },
            },
            {
                "model_id": "model-a",
                "task_id": "task-2",
                "prompt_variant_id": "direct",
                "run_index": 0,
                "temperature": 0.0,
                "status": "completed",
                "metric": "exact_match",
                "score": 0.0,
                "scores": {
                    "exact_match": 0.0,
                    "normalized_code_match": 0.0,
                    "syntax_check": 1.0,
                },
            },
        ]
    )


class TestPrimaryMetricConfig:
    def test_primary_metric_loaded_from_yaml(self, tmp_path: Path) -> None:
        config_dict = {
            "experiment_id": "demo",
            "models": [{"id": "m1", "provider": "mock", "model_id": "mock-v1"}],
            "providers": {"mock": {"type": "mock"}},
            "tasks": [{"id": "t1", "dataset": "data.jsonl", "domain": "code_generation"}],
            "evaluation_metrics": ["exact_match", "normalized_code_match"],
            "primary_metric": "normalized_code_match",
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(config_dict), encoding="utf-8")

        config = load_config(path)
        assert config.primary_metric == "normalized_code_match"
        assert resolve_primary_metric(config) == ("normalized_code_match", [])

    def test_missing_primary_metric_emits_warning(self) -> None:
        config = ExperimentConfig.model_validate(
            {
                "experiment_id": "demo",
                "models": [{"id": "m1", "provider": "mock", "model_id": "mock-v1"}],
                "providers": {"mock": {"type": "mock"}},
                "tasks": [{"id": "t1", "dataset": "data.jsonl"}],
                "evaluation_metrics": ["exact_match", "normalized_code_match"],
            }
        )
        metric, warnings = resolve_primary_metric(config)
        assert metric == "exact_match"
        assert warnings
        assert "primary_metric not configured" in warnings[0]

    def test_invalid_primary_metric_rejected(self) -> None:
        with pytest.raises(ValueError, match="primary_metric"):
            ExperimentConfig.model_validate(
                {
                    "experiment_id": "demo",
                    "models": [{"id": "m1", "provider": "mock", "model_id": "mock-v1"}],
                    "providers": {"mock": {"type": "mock"}},
                    "tasks": [{"id": "t1", "dataset": "data.jsonl"}],
                    "evaluation_metrics": ["exact_match"],
                    "primary_metric": "normalized_code_match",
                }
            )


class TestPrepareResultsPrimaryMetric:
    def test_extracts_primary_metric_from_scores(self, code_results_df: pd.DataFrame) -> None:
        prepared = prepare_results_table(
            code_results_df,
            metric_name="normalized_code_match",
        )
        assert prepared["metric_name"].tolist() == ["normalized_code_match", "normalized_code_match"]
        assert prepared["metric_value"].tolist() == [1.0, 0.0]

    def test_grouped_stats_change_when_metric_changes(self, code_results_df: pd.DataFrame) -> None:
        exact = prepare_results_table(code_results_df, metric_name="exact_match")
        normalized = prepare_results_table(code_results_df, metric_name="normalized_code_match")

        assert exact["metric_value"].mean() == 0.0
        assert normalized["metric_value"].mean() == 0.5


class TestReportPrimaryMetric:
    def test_report_uses_primary_metric_in_descriptive_stats(
        self,
        tmp_path: Path,
        code_results_df: pd.DataFrame,
    ) -> None:
        config = ExperimentConfig.model_validate(
            {
                "experiment_id": "demo",
                "models": [{"id": "model-a", "provider": "mock", "model_id": "mock-v1"}],
                "providers": {"mock": {"type": "mock"}},
                "tasks": [{"id": "task-1", "dataset": "data.jsonl"}],
                "evaluation_metrics": ["exact_match", "normalized_code_match", "syntax_check"],
                "primary_metric": "normalized_code_match",
            }
        )
        manifest = {
            "status": "completed",
            "run_id": "run1",
            "started_at": "",
            "finished_at": "",
            "execution_duration_seconds": 1.0,
            "total_cells": 2,
            "completed_cells": 2,
            "failed_cells": 0,
            "skipped_cells": 0,
        }
        report_path = tmp_path / "report.md"
        generate_report(
            config=config,
            manifest=manifest,
            results_df=code_results_df,
            output_path=report_path,
        )
        report = report_path.read_text(encoding="utf-8")

        assert "**Primary metric**: normalized_code_match" in report
        assert "| model-a | 2 | 0.5 |" in report

    def test_output_quality_warning_when_exact_zero_normalized_positive(
        self,
        tmp_path: Path,
        code_results_df: pd.DataFrame,
    ) -> None:
        config = ExperimentConfig.model_validate(
            {
                "experiment_id": "demo",
                "models": [{"id": "model-a", "provider": "mock", "model_id": "mock-v1"}],
                "providers": {"mock": {"type": "mock"}},
                "tasks": [{"id": "task-1", "dataset": "data.jsonl"}],
                "evaluation_metrics": ["exact_match", "normalized_code_match"],
                "primary_metric": "normalized_code_match",
            }
        )
        manifest = {
            "status": "completed",
            "run_id": "run1",
            "started_at": "",
            "finished_at": "",
            "execution_duration_seconds": 1.0,
            "total_cells": 2,
            "completed_cells": 2,
            "failed_cells": 0,
            "skipped_cells": 0,
        }
        report_path = tmp_path / "report.md"
        generate_report(
            config=config,
            manifest=manifest,
            results_df=code_results_df,
            output_path=report_path,
        )
        report = report_path.read_text(encoding="utf-8")

        assert "## Output quality note" in report
        assert "Strict exact_match is zero" in report

    def test_default_primary_metric_warning_in_report(
        self,
        tmp_path: Path,
        code_results_df: pd.DataFrame,
    ) -> None:
        config = ExperimentConfig.model_validate(
            {
                "experiment_id": "demo",
                "models": [{"id": "model-a", "provider": "mock", "model_id": "mock-v1"}],
                "providers": {"mock": {"type": "mock"}},
                "tasks": [{"id": "task-1", "dataset": "data.jsonl"}],
                "evaluation_metrics": ["exact_match", "normalized_code_match"],
            }
        )
        manifest = {
            "status": "completed",
            "run_id": "run1",
            "started_at": "",
            "finished_at": "",
            "execution_duration_seconds": 1.0,
            "total_cells": 2,
            "completed_cells": 2,
            "failed_cells": 0,
            "skipped_cells": 0,
        }
        report_path = tmp_path / "report.md"
        generate_report(
            config=config,
            manifest=manifest,
            results_df=code_results_df,
            output_path=report_path,
        )
        report = report_path.read_text(encoding="utf-8")

        assert "**Primary metric**: exact_match" in report
        assert "primary_metric not configured" in report


class TestAnalysisMetricResolution:
    def test_resolve_from_config_beside_results(self, tmp_path: Path) -> None:
        config_dict = {
            "experiment_id": "demo",
            "models": [{"id": "m1", "provider": "mock", "model_id": "mock-v1"}],
            "providers": {"mock": {"type": "mock"}},
            "tasks": [{"id": "t1", "dataset": "data.jsonl"}],
            "evaluation_metrics": ["exact_match", "normalized_code_match"],
            "primary_metric": "normalized_code_match",
        }
        exp_dir = tmp_path / "demo_exp"
        exp_dir.mkdir()
        (exp_dir / "config.yaml").write_text(yaml.dump(config_dict), encoding="utf-8")
        results_path = exp_dir / "results.parquet"
        pd.DataFrame({"score": [0.0]}).to_parquet(results_path)

        metric, warnings = resolve_analysis_metric(
            metric=None,
            results_path=results_path,
        )
        assert metric == "normalized_code_match"
        assert warnings == []
