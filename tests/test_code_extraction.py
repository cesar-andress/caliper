"""Tests for Python code extraction and code-aware metrics."""

from __future__ import annotations

import pandas as pd

from caliper.evaluation.code_extraction import extract_python_code, normalize_code
from caliper.evaluation.code_metrics import normalized_code_match, syntax_check
from caliper.evaluation.context import EvaluationInput
from caliper.evaluation.inspect_output import format_inspection, inspect_experiment, metric_means_from_results


class TestCodeExtraction:
    def test_extract_fenced_python_block(self) -> None:
        text = "Here is the solution:\n```python\ndef add(a, b):\n    return a + b\n```\n"
        assert extract_python_code(text) == "def add(a, b):\n    return a + b"

    def test_extract_generic_fenced_block(self) -> None:
        text = "```\ndef subtract(a, b):\n    return a - b\n```"
        assert extract_python_code(text) == "def subtract(a, b):\n    return a - b"

    def test_extract_raw_def_block(self) -> None:
        text = "Sure!\ndef multiply(a, b):\n    return a * b"
        assert extract_python_code(text).startswith("def multiply")

    def test_extract_fallback_to_original(self) -> None:
        text = "return 42"
        assert extract_python_code(text) == "return 42"

    def test_normalize_code_collapses_blank_lines(self) -> None:
        raw = "```python\ndef add(a, b):\n\n    return a + b\n\n```"
        expected = "def add(a, b):\n\n    return a + b"
        assert normalize_code(raw) == expected


class TestCodeMetrics:
    def test_normalized_code_match_with_fenced_prediction(self) -> None:
        sample = EvaluationInput(
            prediction="```python\ndef add(a, b):\n    return a + b\n```",
            expected_output="def add(a, b):\n    return a + b",
            domain="code_generation",
        )
        result = normalized_code_match(sample)
        assert result.value == 1.0
        assert result.success is True

    def test_normalized_code_match_failure(self) -> None:
        sample = EvaluationInput(
            prediction="```python\ndef add(a, b):\n    return 0\n```",
            expected_output="def add(a, b):\n    return a + b",
            domain="code_generation",
        )
        result = normalized_code_match(sample)
        assert result.value == 0.0

    def test_syntax_check_pass(self) -> None:
        sample = EvaluationInput(
            prediction="```python\ndef add(a, b):\n    return a + b\n```",
            domain="code_generation",
        )
        result = syntax_check(sample)
        assert result.value == 1.0
        assert result.success is True

    def test_syntax_check_fail(self) -> None:
        sample = EvaluationInput(
            prediction="```python\ndef broken(:\n    pass\n```",
            domain="code_generation",
        )
        result = syntax_check(sample)
        assert result.value == 0.0
        assert result.success is False
        assert "syntax_error" in result.metadata


class TestInspectOutput:
    def test_metric_means_from_scores_dict(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "status": "completed",
                    "scores": {
                        "exact_match": 0.0,
                        "normalized_code_match": 1.0,
                        "syntax_check": 1.0,
                    },
                },
                {
                    "status": "completed",
                    "scores": {
                        "exact_match": 0.0,
                        "normalized_code_match": 0.5,
                        "syntax_check": 1.0,
                    },
                },
            ]
        )
        means = metric_means_from_results(df)
        assert means["exact_match"] == 0.0
        assert means["normalized_code_match"] == 0.75
        assert means["syntax_check"] == 1.0

    def test_format_inspection_renders_fields(self) -> None:
        rendered = format_inspection(
            [
                {
                    "task_id": "task-pilot-001",
                    "prompt_id": "direct",
                    "model": "qwen25_coder_7b",
                    "prediction": "def add(a, b):\n    return a + b",
                    "expected_output": "def add(a, b):\n    return a + b",
                    "metrics": {
                        "exact_match": 1.0,
                        "normalized_code_match": 1.0,
                        "syntax_check": 1.0,
                    },
                }
            ]
        )
        assert "task-pilot-001" in rendered
        assert "normalized_code_match: 1.0" in rendered

    def test_inspect_experiment_reads_results(self, tmp_path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        df = pd.DataFrame(
            [
                {
                    "cell_id": "abc",
                    "status": "completed",
                    "task_id": "task-pilot-001",
                    "prompt_variant_id": "direct",
                    "model_id": "qwen25_coder_7b",
                    "prediction": "```python\ndef add(a, b):\n    return a + b\n```",
                    "scores": {
                        "exact_match": 0.0,
                        "normalized_code_match": 1.0,
                        "syntax_check": 1.0,
                    },
                }
            ]
        )
        pq.write_table(pa.Table.from_pydict({col: df[col].tolist() for col in df.columns}), tmp_path / "results.parquet")

        eval_df = pd.DataFrame(
            [
                {
                    "cell_id": "abc",
                    "expected_output": "def add(a, b):\n    return a + b",
                    "metric_name": "normalized_code_match",
                    "metric_value": 1.0,
                }
            ]
        )
        pq.write_table(
            pa.Table.from_pydict({col: eval_df[col].tolist() for col in eval_df.columns}),
            tmp_path / "evaluations.parquet",
        )

        records = inspect_experiment(tmp_path, limit=1)
        assert len(records) == 1
        assert records[0]["metrics"]["normalized_code_match"] == 1.0
