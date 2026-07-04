"""Tests for evaluation metrics and post-hoc runner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from caliper.config.loader import load_config
from caliper.evaluation.code_metrics import contains_expected, evaluate_test_pass, exact_match
from caliper.evaluation.context import EvaluationInput
from caliper.evaluation.metrics import compute_metric
from caliper.evaluation.registry import EvaluationOptions, evaluate_sample, metrics_for_domain
from caliper.evaluation.runner import evaluate_results_file
from caliper.evaluation.summarization_metrics import length, lexical_overlap, llm_judge

EXAMPLE_FACTORIAL = Path("configs/examples/example_factorial.yaml")


class TestCodeMetrics:
    def test_exact_match_success(self) -> None:
        sample = EvaluationInput(
            prediction="def add(a, b):\n    return a + b",
            expected_output="def add(a, b):\n    return a + b",
            domain="code_generation",
        )
        result = exact_match(sample)
        assert result.name == "exact_match"
        assert result.value == 1.0
        assert result.success is True

    def test_exact_match_failure(self) -> None:
        sample = EvaluationInput(
            prediction="def add(a, b): return 0",
            expected_output="def add(a, b):\n    return a + b",
            domain="code_generation",
        )
        result = exact_match(sample)
        assert result.value == 0.0
        assert result.success is False

    def test_exact_match_no_reference(self) -> None:
        result = exact_match(EvaluationInput(prediction="x", domain="code_generation"))
        assert result.success is False
        assert "reason" in result.metadata

    def test_contains_expected_success(self) -> None:
        sample = EvaluationInput(
            prediction="Here is the code:\ndef add(a, b):\n    return a + b\nDone.",
            expected_output="def add(a, b):\n    return a + b",
            domain="code_generation",
        )
        result = contains_expected(sample)
        assert result.value == 1.0
        assert result.success is True

    def test_contains_expected_failure(self) -> None:
        sample = EvaluationInput(
            prediction="return 0",
            expected_output="def add(a, b):\n    return a + b",
            domain="code_generation",
        )
        result = contains_expected(sample)
        assert result.value == 0.0
        assert result.success is False

    def test_test_pass_disabled_by_default(self) -> None:
        sample = EvaluationInput(
            prediction="def add(a,b): return a+b",
            tests=["assert add(1,2)==3"],
            domain="bug_repair",
        )
        result = evaluate_test_pass(sample, execution_enabled=False)
        assert result.name == "test_pass"
        assert result.success is False
        assert result.metadata["status"] == "disabled"

    def test_test_pass_enabled_placeholder(self) -> None:
        sample = EvaluationInput(
            prediction="def add(a,b): return a+b",
            tests=["assert add(1,2)==3"],
            domain="bug_repair",
        )
        result = evaluate_test_pass(sample, execution_enabled=True)
        assert result.metadata["status"] == "not_implemented"

    def test_test_pass_no_tests(self) -> None:
        result = evaluate_test_pass(
            EvaluationInput(prediction="x", domain="bug_repair"),
            execution_enabled=True,
        )
        assert result.metadata["status"] == "skipped"


class TestSummarizationMetrics:
    def test_length(self) -> None:
        result = length(EvaluationInput(prediction="Computes fibonacci recursively.", domain="code_summarization"))
        assert result.name == "length"
        assert result.value == 3.0
        assert result.success is True
        assert result.metadata["word_count"] == 3

    def test_length_empty_prediction(self) -> None:
        result = length(EvaluationInput(prediction="   ", domain="code_summarization"))
        assert result.value == 0.0
        assert result.success is False

    def test_lexical_overlap_high(self) -> None:
        sample = EvaluationInput(
            prediction="Computes the nth Fibonacci number recursively.",
            expected_output="Computes the nth Fibonacci number using recursion.",
            domain="code_summarization",
        )
        result = lexical_overlap(sample)
        assert result.value > 0.5
        assert result.success is True
        assert "precision" in result.metadata

    def test_lexical_overlap_no_reference(self) -> None:
        result = lexical_overlap(EvaluationInput(prediction="hello", domain="code_summarization"))
        assert result.success is False

    def test_llm_judge_disabled(self) -> None:
        result = llm_judge(EvaluationInput(prediction="summary", domain="code_summarization"))
        assert result.metadata["status"] == "disabled"
        assert result.success is False

    def test_llm_judge_enabled_placeholder(self) -> None:
        result = llm_judge(
            EvaluationInput(prediction="summary", domain="code_summarization"),
            enabled=True,
        )
        assert result.metadata["status"] == "not_implemented"


class TestRegistry:
    def test_code_domain_metrics(self) -> None:
        assert metrics_for_domain("code_generation") == [
            "exact_match",
            "contains_expected",
            "test_pass",
        ]
        assert metrics_for_domain("bug_repair") == [
            "exact_match",
            "contains_expected",
            "test_pass",
        ]

    def test_summarization_domain_metrics(self) -> None:
        assert metrics_for_domain("code_summarization") == [
            "length",
            "lexical_overlap",
            "llm_judge",
        ]

    def test_evaluate_sample_code(self) -> None:
        sample = EvaluationInput(
            prediction="def add(a, b):\n    return a + b",
            expected_output="def add(a, b):\n    return a + b",
            tests=["assert add(1,2)==3"],
            domain="code_generation",
        )
        metrics = evaluate_sample(sample)
        assert len(metrics) == 3
        assert all(hasattr(m, "name") and hasattr(m, "value") and hasattr(m, "success") for m in metrics)

    def test_evaluate_sample_summarization(self) -> None:
        sample = EvaluationInput(
            prediction="Recursive fibonacci function.",
            expected_output="Computes fibonacci recursively.",
            domain="code_summarization",
        )
        metrics = evaluate_sample(sample)
        assert len(metrics) == 3
        assert metrics[0].name == "length"


class TestComputeMetricBatch:
    def test_batch_exact_match(self) -> None:
        result = compute_metric(
            "exact_match",
            ["a", "b"],
            ["a", "c"],
        )
        assert result.value == 0.5
        assert result.metadata["n"] == 2


class TestEvaluateResultsFile:
    def test_evaluate_factorial_results(self, tmp_path: Path) -> None:
        from caliper.runners.experiment import ExperimentRunner

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

        results_path = runner.output_dir / "results.parquet"
        summary = evaluate_results_file(
            results_path,
            loaded,
            config_path=config_path,
        )

        assert summary["rows_evaluated"] == 1
        assert Path(summary["output_parquet"]).exists()
        assert Path(summary["output_jsonl"]).exists()

        eval_df = pd.read_parquet(summary["output_parquet"])
        assert len(eval_df) == 3  # exact_match, contains_expected, test_pass
        assert set(eval_df["metric_name"]) == {"exact_match", "contains_expected", "test_pass"}
        assert "metric_success" in eval_df.columns

    def test_skips_non_completed_rows(self, tmp_path: Path) -> None:
        config = load_config(EXAMPLE_FACTORIAL)
        results_path = tmp_path / "results.parquet"
        pd.DataFrame(
            [
                {
                    "cell_id": "abc",
                    "experiment_id": "example_factorial",
                    "run_id": "r1",
                    "run_index": 0,
                    "model_id": "mock-model",
                    "provider_name": "mock",
                    "provider_type": "mock",
                    "task_id": "cg-sample",
                    "prompt_variant_id": "direct",
                    "temperature": 0.0,
                    "metric": "exact_match",
                    "score": 0.0,
                    "status": "failed",
                    "prediction": "",
                }
            ]
        ).to_parquet(results_path)

        summary = evaluate_results_file(
            results_path,
            config,
            config_path=EXAMPLE_FACTORIAL,
        )
        assert summary["rows_evaluated"] == 0
