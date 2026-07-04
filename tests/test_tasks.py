"""Tests for benchmark task system."""

from __future__ import annotations

from pathlib import Path

import pytest

from caliper.tasks import (
    BugRepairTask,
    CodeGenerationTask,
    CodeSummarizationTask,
    TaskDataset,
    TaskMetadata,
    TaskValidationError,
    create_task,
    list_task_domains,
    validate_dataset,
    validate_dataset_file,
    validate_task_record,
)

DATA_DIR = Path("data/examples")


class TestTaskMetadata:
    def test_valid_record(self) -> None:
        record = validate_task_record(
            {
                "task_id": "cg-001",
                "domain": "code_generation",
                "input": "Write add(a,b)",
                "expected_output": "def add(a,b): return a+b",
                "language": "python",
                "source_benchmark": "humaneval-sample",
                "tags": ["math"],
                "difficulty": "easy",
            }
        )
        assert record.task_id == "cg-001"
        assert record.language == "python"

    def test_missing_required_field(self) -> None:
        with pytest.raises(TaskValidationError, match="task_id"):
            validate_task_record({"domain": "code_generation", "input": "x", "language": "py", "source_benchmark": "b"})

    def test_invalid_task_id(self) -> None:
        with pytest.raises(TaskValidationError):
            validate_task_record(
                {
                    "task_id": "Bad-ID",
                    "domain": "code_generation",
                    "input": "x",
                    "language": "python",
                    "source_benchmark": "b",
                }
            )

    def test_empty_input_rejected(self) -> None:
        with pytest.raises(TaskValidationError, match="input"):
            validate_task_record(
                {
                    "task_id": "cg-bad",
                    "domain": "code_generation",
                    "input": "   ",
                    "language": "python",
                    "source_benchmark": "b",
                }
            )

    def test_domain_mismatch(self) -> None:
        with pytest.raises(TaskValidationError, match="domain"):
            validate_task_record(
                {
                    "task_id": "cg-001",
                    "domain": "bug_repair",
                    "input": "x",
                    "language": "python",
                    "source_benchmark": "b",
                },
                expected_domain="code_generation",
            )


class TestTaskDatasetLoader:
    @pytest.mark.parametrize(
        ("filename", "expected_count"),
        [
            ("code_generation_sample.jsonl", 3),
            ("bug_repair_sample.jsonl", 3),
            ("code_summarization_sample.jsonl", 3),
        ],
    )
    def test_load_example_datasets(self, filename: str, expected_count: int) -> None:
        dataset = TaskDataset.from_jsonl(DATA_DIR / filename)
        assert len(dataset) == expected_count
        assert all(isinstance(r, TaskMetadata) for r in dataset)

    def test_load_with_limit(self) -> None:
        dataset = TaskDataset.from_jsonl(
            DATA_DIR / "code_generation_sample.jsonl",
            limit=2,
        )
        assert len(dataset) == 2

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            TaskDataset.from_jsonl("nonexistent.jsonl")

    def test_invalid_json_line(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text('{"task_id": "x"\n', encoding="utf-8")
        with pytest.raises(TaskValidationError, match="invalid JSON"):
            TaskDataset.from_jsonl(path)

    def test_duplicate_ids_detected(self, tmp_path: Path) -> None:
        path = tmp_path / "dup.jsonl"
        row = (
            '{"task_id": "cg-001", "domain": "code_generation", "input": "a", '
            '"language": "python", "source_benchmark": "b"}\n'
        )
        path.write_text(row + row, encoding="utf-8")
        dataset = TaskDataset.from_jsonl(path)
        errors = validate_dataset(dataset, expected_domain="code_generation")
        assert any("duplicate task_id" in e for e in errors)


class TestValidateDatasetFile:
    def test_example_files_are_valid(self) -> None:
        for name in [
            "code_generation_sample.jsonl",
            "bug_repair_sample.jsonl",
            "code_summarization_sample.jsonl",
        ]:
            errors = validate_dataset_file(DATA_DIR / name)
            assert errors == [], f"{name} validation errors: {errors}"

    def test_domain_check(self) -> None:
        errors = validate_dataset_file(
            DATA_DIR / "bug_repair_sample.jsonl",
            expected_domain="code_generation",
        )
        assert len(errors) == 3


class TestConcreteTasks:
    def test_registry_lists_domains(self) -> None:
        domains = list_task_domains()
        assert "code_generation" in domains
        assert "bug_repair" in domains
        assert "code_summarization" in domains

    def test_code_generation_task_load_and_score(self) -> None:
        task = CodeGenerationTask(
            "cg-eval",
            DATA_DIR / "code_generation_sample.jsonl",
        )
        examples = task.load_examples()
        assert len(examples) == 3
        example = examples[0]
        scores = task.score(example, example.expected_output or "")
        assert scores["exact_match"] == 1.0

    def test_bug_repair_task_score(self) -> None:
        task = BugRepairTask("br-eval", DATA_DIR / "bug_repair_sample.jsonl")
        example = task.load_examples()[0]
        scores = task.score(example, example.expected_output or "")
        assert scores["exact_match"] == 1.0
        assert scores["bug_removed"] == 1.0

    def test_code_summarization_task_score(self) -> None:
        task = CodeSummarizationTask("cs-eval", DATA_DIR / "code_summarization_sample.jsonl")
        example = task.load_examples()[0]
        scores = task.score(example, "Some unrelated summary")
        assert scores["exact_match"] == 0.0
        assert 0.0 <= scores["rouge_l"] <= 1.0

    def test_create_task_factory(self) -> None:
        task = create_task(
            "code_generation",
            "via-factory",
            DATA_DIR / "code_generation_sample.jsonl",
        )
        assert isinstance(task, CodeGenerationTask)
        assert task.num_examples() == 3

    def test_wrong_domain_in_dataset_rejected(self) -> None:
        with pytest.raises(TaskValidationError, match="domain"):
            CodeGenerationTask("bad", DATA_DIR / "bug_repair_sample.jsonl")

    def test_metadata_fields_present(self) -> None:
        task = CodeGenerationTask("cg", DATA_DIR / "code_generation_sample.jsonl")
        record = task.load_examples()[0]
        assert record.task_id
        assert record.domain == "code_generation"
        assert record.input
        assert record.language == "python"
        assert record.source_benchmark
        assert isinstance(record.tags, list)
        assert record.difficulty in ("easy", "medium", "hard", None)

    def test_num_samples_config(self) -> None:
        task = CodeGenerationTask(
            "cg",
            DATA_DIR / "code_generation_sample.jsonl",
            config={"num_samples": 1},
        )
        assert task.num_examples() == 1
