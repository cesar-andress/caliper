"""Materialize official benchmarks to CALIPER JSONL datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from caliper.benchmarks.base import BenchmarkName
from caliper.benchmarks.humaneval_plus import load_humaneval_plus, to_task_metadata
from caliper.benchmarks.mbpp import load_mbpp, mbpp_to_task_metadata
from caliper.tasks.schema import TaskMetadata

DEFAULT_DATA_DIR = Path("data/benchmarks")
BENCHMARK_LOADERS = {
    "humaneval_plus": (load_humaneval_plus, to_task_metadata),
    "mbpp": (lambda: load_mbpp(split="test"), mbpp_to_task_metadata),
}


def _write_jsonl(path: Path, records: list[TaskMetadata]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")


def materialize_benchmark(
    name: BenchmarkName,
    output_dir: Path | str = DEFAULT_DATA_DIR,
    *,
    limit: int | None = None,
) -> Path:
    """Download/parse a benchmark and write a CALIPER-compatible JSONL file."""
    loader, converter = BENCHMARK_LOADERS[name]
    info, benchmark_records = loader()
    tasks = [converter(record) for record in benchmark_records]
    if limit is not None:
        tasks = tasks[: int(limit)]

    out_dir = Path(output_dir)
    out_path = out_dir / f"{name}.jsonl"
    _write_jsonl(out_path, tasks)

    manifest = {
        "benchmark": name,
        "version": info.version,
        "source_url": info.source_url,
        "license": info.license,
        "num_tasks_written": len(tasks),
        "num_tasks_available": info.num_tasks,
    }
    (out_dir / f"{name}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return out_path


def materialize_all(
    output_dir: Path | str = DEFAULT_DATA_DIR,
    *,
    limit: int | None = None,
) -> dict[str, Path]:
    """Materialize all supported confirmatory benchmarks."""
    paths: dict[str, Path] = {}
    for name in BENCHMARK_LOADERS:
        paths[name] = materialize_benchmark(name, output_dir, limit=limit)  # type: ignore[arg-type]
    return paths


def list_all_task_ids(dataset_path: Path | str) -> list[str]:
    """Return all unique task IDs from a JSONL dataset in sorted order."""
    from caliper.tasks.loader import TaskDataset

    dataset = TaskDataset.from_jsonl(dataset_path)
    return sorted(dataset.ids())


def select_task_subset(
    dataset_path: Path | str,
    *,
    size: int,
    seed: int = 20260404,
) -> list[str]:
    """Deterministically select a stratified subset of task ids from a JSONL dataset."""
    from caliper.tasks.loader import TaskDataset

    dataset = TaskDataset.from_jsonl(dataset_path)
    ids = dataset.ids()
    if size >= len(ids):
        return ids

    import random

    rng = random.Random(seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    return sorted(shuffled[:size])
