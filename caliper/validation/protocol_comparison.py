"""Compare confirmatory experiment protocols for Paper 1 extensions."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from caliper.benchmarks.experiment_yaml import expected_cell_count
from caliper.benchmarks.materialize import list_all_task_ids
from caliper.config.loader import load_config

SUBSET_CONFIG = Path("configs/paper1/confirmatory_humaneval.yaml")
FULL_CONFIG = Path("configs/paper1/confirmatory_humaneval_full.yaml")
DATASET_PATH = Path("data/benchmarks/humaneval_plus.jsonl")

PROTOCOL_TOP_LEVEL_KEYS = (
    "random_seed",
    "primary_metric",
    "evaluation_metrics",
    "providers",
    "models",
    "prompt_variants",
    "temperatures",
    "number_of_runs",
    "decoding",
    "logging",
    "execution",
)

STUDY_METADATA_KEYS = (
    "study_type",
    "benchmark",
    "prompt_protocol",
    "pilot_reference",
)


@dataclass
class ProtocolComparisonResult:
    subset_path: Path
    full_path: Path
    passed: bool
    subset_task_count: int
    full_task_count: int
    expected_full_cells: int
    subset_filter_task_ids: list[str] = field(default_factory=list)
    full_filter_task_ids: list[str] = field(default_factory=list)
    differences: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "# Paper 1 HumanEval+ protocol comparison",
            "",
            f"- Subset config: `{self.subset_path}`",
            f"- Full config: `{self.full_path}`",
            f"- Result: **{'PASS' if self.passed else 'FAIL'}**",
            "",
            "## Task coverage",
            "",
            f"| Study | Task slots | Unique benchmark task IDs | Expected cells |",
            f"|-------|------------|---------------------------|----------------|",
            f"| 40-task confirmatory | {self.subset_task_count} | {len(self.subset_filter_task_ids)} | {expected_cell_count(self.subset_task_count)} |",
            f"| 164-task extension | {self.full_task_count} | {len(self.full_filter_task_ids)} | {self.expected_full_cells} |",
            "",
        ]
        if self.notes:
            lines.extend(["## Expected differences", ""])
            lines.extend(f"- {note}" for note in self.notes)
            lines.append("")
        if self.differences:
            lines.extend(["## Unintended differences", ""])
            lines.extend(f"- {item}" for item in self.differences)
            lines.append("")
        else:
            lines.extend(
                [
                    "## Protocol dimensions checked",
                    "",
                    "- Model set",
                    "- Prompt family templates",
                    "- Temperature levels",
                    "- Run count",
                    "- Primary and secondary evaluation metrics",
                    "- Provider configuration",
                    "- Sandbox execution settings (timeout, memory)",
                    "- Random seed policy",
                    "- Decoding parameters",
                    "- Execution settings (shuffle, workers)",
                    "- Logging settings",
                    "",
                    "Only task coverage differs between the two configurations.",
                    "",
                ]
            )
        return "\n".join(lines)


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _filter_task_ids(config: dict[str, Any]) -> list[str]:
    tasks = config.get("tasks") or []
    ids: list[str] = []
    for task in tasks:
        extra = task.get("extra") or {}
        benchmark_id = extra.get("filter_task_id")
        if benchmark_id:
            ids.append(str(benchmark_id))
    return sorted(ids)


def _sandbox_signature(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    if not tasks:
        return {}
    sample = tasks[0].get("extra") or {}
    execution = sample.get("execution") or {}
    return {
        "timeout_seconds": execution.get("timeout_seconds"),
        "memory_mb": execution.get("memory_mb"),
        "domain": tasks[0].get("domain"),
        "dataset": tasks[0].get("dataset"),
        "metrics": tasks[0].get("metrics"),
        "num_samples": tasks[0].get("num_samples"),
    }


def _protocol_view(config: dict[str, Any]) -> dict[str, Any]:
    view: dict[str, Any] = {}
    for key in PROTOCOL_TOP_LEVEL_KEYS:
        if key in config:
            view[key] = copy.deepcopy(config[key])
    metadata = config.get("study_metadata") or {}
    view["study_metadata"] = {
        key: metadata[key] for key in STUDY_METADATA_KEYS if key in metadata
    }
    view["sandbox"] = _sandbox_signature(config.get("tasks") or [])
    return view


def _compare_section(name: str, left: Any, right: Any) -> list[str]:
    if left == right:
        return []
    return [f"{name}: subset={left!r} full={right!r}"]


def compare_protocols(
    subset_path: Path | str = SUBSET_CONFIG,
    full_path: Path | str = FULL_CONFIG,
    *,
    dataset_path: Path | str = DATASET_PATH,
    expected_full_tasks: int = 164,
    expected_full_cells: int = 39_360,
) -> ProtocolComparisonResult:
    subset_path = Path(subset_path)
    full_path = Path(full_path)
    dataset_path = Path(dataset_path)

    subset = _load_yaml_dict(subset_path)
    full = _load_yaml_dict(full_path)

    subset_ids = _filter_task_ids(subset)
    full_ids = _filter_task_ids(full)
    differences: list[str] = []
    notes = [
        "experiment_id differs by design",
        "description differs by design",
        "output.directory differs by design",
        f"task count: {len(subset_ids)} vs {len(full_ids)}",
    ]

    differences.extend(_compare_section("protocol", _protocol_view(subset), _protocol_view(full)))

    if len(full_ids) != expected_full_tasks:
        differences.append(
            f"full task count {len(full_ids)} != expected {expected_full_tasks}"
        )
    if len(set(full_ids)) != len(full_ids):
        differences.append("full config contains duplicate benchmark task IDs")

    available_ids = list_all_task_ids(dataset_path)
    if set(full_ids) != set(available_ids):
        missing = sorted(set(available_ids) - set(full_ids))
        extra = sorted(set(full_ids) - set(available_ids))
        if missing:
            differences.append(f"full config missing benchmark IDs: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if extra:
            differences.append(f"full config has unknown benchmark IDs: {extra[:5]}{'...' if len(extra) > 5 else ''}")

    if not set(subset_ids).issubset(set(full_ids)):
        differences.append("40-task subset is not contained in the 164-task full configuration")

    computed_cells = expected_cell_count(len(full_ids))
    if computed_cells != expected_full_cells:
        differences.append(
            f"expected cell count mismatch: computed {computed_cells}, required {expected_full_cells}"
        )

    passed = not differences
    return ProtocolComparisonResult(
        subset_path=subset_path,
        full_path=full_path,
        passed=passed,
        subset_task_count=len(subset_ids),
        full_task_count=len(full_ids),
        expected_full_cells=expected_full_cells,
        subset_filter_task_ids=subset_ids,
        full_filter_task_ids=full_ids,
        differences=differences,
        notes=notes,
    )


def write_protocol_comparison_report(
    output_path: Path | str,
    *,
    subset_path: Path | str = SUBSET_CONFIG,
    full_path: Path | str = FULL_CONFIG,
) -> ProtocolComparisonResult:
    result = compare_protocols(subset_path=subset_path, full_path=full_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.to_markdown(), encoding="utf-8")
    sidecar = output_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "passed": result.passed,
                "differences": result.differences,
                "subset_task_count": result.subset_task_count,
                "full_task_count": result.full_task_count,
                "expected_full_cells": result.expected_full_cells,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def assert_protocol_equivalent_except_tasks(
    subset_path: Path | str = SUBSET_CONFIG,
    full_path: Path | str = FULL_CONFIG,
) -> ProtocolComparisonResult:
    result = compare_protocols(subset_path=subset_path, full_path=full_path)
    if not result.passed:
        detail = "\n".join(result.differences)
        msg = f"Protocol comparison failed:\n{detail}"
        raise ValueError(msg)
    return result
