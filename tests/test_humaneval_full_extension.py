"""Tests for the HumanEval+ full-benchmark confirmatory extension."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from caliper.benchmarks.experiment_yaml import expected_cell_count, write_humaneval_full_config
from caliper.benchmarks.materialize import list_all_task_ids
from caliper.config.loader import load_config, validate_config
from caliper.runners.checkpoint import CheckpointStore
from caliper.runners.experiment import ExperimentRunner
from caliper.runners.experiment_paths import resolve_experiment_dir
from caliper.runners.results import ExperimentResultRecord
from caliper.statistics.design_guidance import build_design_recommendations, export_design_guidance
from caliper.statistics.task_sampling import (
    compare_point_estimates,
    simulate_task_subsets,
    summarize_reliability_thresholds,
)
from caliper.validation.protocol_comparison import compare_protocols

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBSET_CONFIG = REPO_ROOT / "configs/paper1/confirmatory_humaneval.yaml"
FULL_CONFIG = REPO_ROOT / "configs/paper1/confirmatory_humaneval_full.yaml"
DATASET = REPO_ROOT / "data/benchmarks/humaneval_plus.jsonl"
SUBSET_EXPERIMENT = (
    REPO_ROOT / "experiments/paper1_confirmatory_humaneval/paper1_confirmatory_humaneval"
)


@pytest.fixture(scope="module")
def full_config_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if FULL_CONFIG.exists():
        return FULL_CONFIG
    tmp = tmp_path_factory.mktemp("configs")
    output = tmp / "confirmatory_humaneval_full.yaml"
    write_humaneval_full_config(
        reference_path=SUBSET_CONFIG,
        dataset_path=DATASET,
        output_path=output,
    )
    return output


class TestHumanevalFullConfig:
    def test_full_config_validates(self, full_config_path: Path) -> None:
        errors = validate_config(full_config_path)
        assert errors == []
        config = load_config(full_config_path)
        assert config.experiment_id == "paper1_confirmatory_humaneval_full"

    def test_expected_cell_count_is_39360(self, full_config_path: Path) -> None:
        config = load_config(full_config_path)
        planned = len(ExperimentRunner(config, dry_run=True).plan_combinations())
        assert planned == 39_360
        assert expected_cell_count(len(config.tasks)) == 39_360

    def test_exactly_164_unique_tasks(self, full_config_path: Path) -> None:
        config = load_config(full_config_path)
        filter_ids = [
            task.extra.get("filter_task_id")
            for task in config.tasks
            if task.extra and task.extra.get("filter_task_id")
        ]
        assert len(filter_ids) == 164
        assert len(set(filter_ids)) == 164
        assert len(list_all_task_ids(DATASET)) == 164


class TestProtocolComparison:
    def test_accepts_task_count_only_difference(self, full_config_path: Path) -> None:
        result = compare_protocols(
            subset_path=SUBSET_CONFIG,
            full_path=full_config_path,
        )
        assert result.passed
        assert result.subset_task_count == 40
        assert result.full_task_count == 164

    def test_detects_unintended_protocol_difference(self, full_config_path: Path, tmp_path: Path) -> None:
        broken = yaml.safe_load(full_config_path.read_text(encoding="utf-8"))
        broken["temperatures"] = [0.0, 0.7]
        broken_path = tmp_path / "broken.yaml"
        broken_path.write_text(yaml.safe_dump(broken), encoding="utf-8")
        result = compare_protocols(subset_path=SUBSET_CONFIG, full_path=broken_path)
        assert not result.passed
        assert any("temperatures" in diff for diff in result.differences)


class TestTaskSamplingSynthetic:
    def test_task_sampling_analysis_on_synthetic(self) -> None:
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(0)
        rows = []
        tasks = [f"task-{index:03d}" for index in range(1, 21)]
        models = ["m1", "m2", "m3"]
        prompts = ["p1", "p2"]
        for task in tasks:
            for model in models:
                for prompt in prompts:
                    for run in range(2):
                        rows.append(
                            {
                                "task_id": task,
                                "model": model,
                                "prompt_id": prompt,
                                "run_id": run,
                                "temperature": 0.0,
                                "metric_value": float(rng.random()),
                                "pass_fail": int(rng.random() > 0.5),
                            }
                        )
        full = pd.DataFrame(rows)
        subset = full[full["task_id"].isin(tasks[:8])].copy()
        comparison = compare_point_estimates(subset, full)
        assert not comparison.empty
        simulation = simulate_task_subsets(full, task_counts=[5, 10], n_subsets=20, seed=1)
        assert not simulation.empty
        recommendations = summarize_reliability_thresholds(simulation)
        assert not recommendations.empty

    def test_ranking_convergence_stable_vs_unstable(self) -> None:
        import pandas as pd
        from scipy.stats import kendalltau

        stable_a = pd.Series({"m1": 0.9, "m2": 0.7, "m3": 0.4})
        stable_b = pd.Series({"m1": 0.88, "m2": 0.69, "m3": 0.41})
        unstable = pd.Series({"m1": 0.4, "m2": 0.9, "m3": 0.7})
        tau_stable, _ = kendalltau(stable_a.rank(), stable_b.rank())
        tau_unstable, _ = kendalltau(stable_a.rank(), unstable.rank())
        assert tau_stable > tau_unstable


class TestDesignGuidance:
    def test_placeholder_export_without_results(self, tmp_path: Path) -> None:
        paths = export_design_guidance(tmp_path / "future_experiment")
        table = build_design_recommendations(tmp_path / "future_experiment")
        assert paths.root.exists()
        assert (paths.root / "table_design_recommendations.csv").exists()
        assert table["status"].eq("pending_experiment_completion").any()

    def test_dstudy_recommendation_extraction_on_subset_experiment(self) -> None:
        if not (SUBSET_EXPERIMENT / "statistical_dataset.parquet").exists():
            pytest.skip("subset experiment outputs unavailable")
        table = build_design_recommendations(SUBSET_EXPERIMENT)
        assert not table.empty
        assert "minimum_tasks_for_g_0_80" in table["recommendation"].values


class TestExperimentPaths:
    def test_resolve_nested_experiment_directory(self) -> None:
        if not SUBSET_EXPERIMENT.exists():
            pytest.skip("subset experiment unavailable")
        resolved = resolve_experiment_dir(SUBSET_EXPERIMENT.parent)
        assert resolved.name == "paper1_confirmatory_humaneval"


class TestResumeAndFailures:
    def test_failures_jsonl_schema_roundtrip(self, tmp_path: Path) -> None:
        from caliper.runners.failures import FailureWriter

        writer = FailureWriter(tmp_path)
        record = ExperimentResultRecord(
            cell_id="cell-1",
            experiment_id="test",
            run_id="run-1",
            run_index=0,
            model_id="m1",
            provider_name="ollama_local",
            provider_type="ollama",
            task_id="t1",
            prompt_variant_id="p1",
            temperature=0.0,
            metric="pass_at_1",
            score=0.0,
            status="failed",
            error="timeout",
        )
        writer.append(record)
        payload = json.loads((tmp_path / "failures.jsonl").read_text(encoding="utf-8").strip())
        assert payload["cell_id"] == "cell-1"

    def test_resume_skips_completed_cells(self, tmp_path: Path) -> None:
        store = CheckpointStore(tmp_path / "checkpoints")
        record = ExperimentResultRecord(
            cell_id="completed-cell",
            experiment_id="paper1_confirmatory_humaneval_full",
            run_id="run-1",
            run_index=0,
            model_id="qwen25_coder_7b",
            provider_name="ollama_local",
            provider_type="ollama",
            task_id="task-humaneval_plus-001",
            prompt_variant_id="minimal",
            temperature=0.0,
            metric="pass_at_1",
            score=1.0,
            status="completed",
        )
        store.write(record)
        assert "completed-cell" in store.load_completed_cell_ids()


class TestExperimentStatus:
    def test_experiment_status_on_subset_experiment(self) -> None:
        if not SUBSET_EXPERIMENT.exists():
            pytest.skip("subset experiment unavailable")
        from caliper.runners.experiment_status import collect_experiment_status

        status = collect_experiment_status(SUBSET_EXPERIMENT)
        assert status.expected_cells == 9_600
        assert status.completed_cells == 9_600
        assert status.percent_complete == 100.0
