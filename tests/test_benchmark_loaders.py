"""Tests for official benchmark loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from caliper.benchmarks.humaneval_plus import to_task_metadata
from caliper.benchmarks.mbpp import mbpp_to_task_metadata
from caliper.benchmarks.base import BenchmarkRecord
from caliper.tasks.loader import TaskDataset


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "benchmarks" / "sample_tasks.jsonl"


def test_fixture_dataset_loads_executable_tasks():
    dataset = TaskDataset.from_jsonl(FIXTURE_PATH)
    assert len(dataset) == 3
    assert all(record.domain == "executable_code_generation" for record in dataset.records)


def test_humaneval_record_to_task_metadata():
    record = BenchmarkRecord(
        benchmark_id="he-test-001",
        prompt="def foo():\n",
        canonical_solution="    return 1\n",
        tests=["def check(c): pass"],
        language="python",
        source_benchmark="humaneval-plus:test",
        harness="humaneval",
        metadata={"entry_point": "foo", "dataset_version": "test"},
    )
    task = to_task_metadata(record)
    assert task.task_id == "he-test-001"
    assert task.extra["harness"] == "humaneval"
    assert task.extra["entry_point"] == "foo"


def test_mbpp_record_to_task_metadata():
    record = BenchmarkRecord(
        benchmark_id="mbpp-0001",
        prompt="Write abs.",
        canonical_solution="def abs_value(n): return n",
        tests=["assert abs_value(1) == 1"],
        language="python",
        source_benchmark="mbpp:test",
        harness="mbpp",
        metadata={"dataset_version": "test"},
    )
    task = mbpp_to_task_metadata(record)
    assert task.domain == "executable_code_generation"
    assert task.extra["harness"] == "mbpp"


@pytest.mark.integration
def test_materialize_humaneval_plus_smoke():
    from caliper.benchmarks.materialize import materialize_benchmark

    path = materialize_benchmark("humaneval_plus", limit=5)
    dataset = TaskDataset.from_jsonl(path)
    assert len(dataset) == 5
