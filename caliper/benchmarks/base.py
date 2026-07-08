"""Shared types for official benchmark loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

BenchmarkName = Literal["humaneval_plus", "mbpp"]
HarnessType = Literal["humaneval", "mbpp"]


@dataclass(frozen=True)
class BenchmarkInfo:
    """Version and provenance metadata for a benchmark snapshot."""

    name: BenchmarkName
    version: str
    source_url: str
    num_tasks: int
    license: str = "unknown"


@dataclass
class BenchmarkRecord:
    """Normalized benchmark instance before conversion to TaskMetadata."""

    benchmark_id: str
    prompt: str
    canonical_solution: str
    tests: list[str]
    language: str
    source_benchmark: str
    harness: HarnessType
    difficulty: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
