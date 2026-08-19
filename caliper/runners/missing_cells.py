"""Inspect factorial coverage and identify missing or unexpected cells."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from caliper.config.schema import ExperimentCombination, ExperimentConfig, PromptVariantConfig
from caliper.runners.cells import expand_cells, make_cell_id
from caliper.runners.checkpoint import CheckpointStore
from caliper.runners.executor import build_task, render_task_prompt
from caliper.runners.failures import (
    duplicate_cell_ids,
    latest_records_by_cell,
    load_results_records,
)
from caliper.runners.reproducibility import cell_seed


@dataclass(frozen=True)
class ExpectedCellSpec:
    """Fully specified factorial cell used for diagnostics and retry."""

    cell_id: str
    model_id: str
    provider: str
    task_id: str
    prompt_variant_id: str
    temperature: float
    run_index: int
    seed: int
    rendered_prompt_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def hash_prompt_text(prompt_text: str) -> str:
    """Return a stable SHA-256 hash of rendered prompt text."""
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def _resolve_prompts(
    config: ExperimentConfig,
    config_dir: Path,
) -> dict[str, PromptVariantConfig]:
    prompt_variants = config.prompt_variants or [
        PromptVariantConfig(id="default", template="{input}")
    ]
    prompts: dict[str, PromptVariantConfig] = {}
    for prompt in prompt_variants:
        if prompt.path is not None and not prompt.path.is_absolute():
            prompts[prompt.id] = prompt.model_copy(
                update={"path": config_dir / prompt.path}
            )
        else:
            prompts[prompt.id] = prompt
    return prompts


def build_expected_cell_specs(
    config: ExperimentConfig,
    *,
    config_dir: Path,
) -> list[ExpectedCellSpec]:
    """Expand the factorial design and enrich each cell with prompt hashes."""
    prompts = _resolve_prompts(config, config_dir)
    tasks = {
        task.id: build_task(config, task, config_dir)
        for task in config.tasks
    }
    specs: list[ExpectedCellSpec] = []
    for cell in expand_cells(config):
        task = tasks[cell.task_id]
        prompt_cfg = prompts[cell.prompt_variant_id]
        examples = task.load_examples()
        if not examples:
            msg = f"task '{cell.task_id}' produced no examples for prompt hashing"
            raise ValueError(msg)
        rendered_prompt = render_task_prompt(prompt_cfg, examples[0], config_dir)
        specs.append(
            ExpectedCellSpec(
                cell_id=make_cell_id(config, cell),
                model_id=cell.model_id,
                provider=cell.provider,
                task_id=cell.task_id,
                prompt_variant_id=cell.prompt_variant_id,
                temperature=cell.temperature,
                run_index=cell.run_index,
                seed=cell_seed(config, cell),
                rendered_prompt_hash=hash_prompt_text(rendered_prompt),
            )
        )
    return specs



def load_observed_cell_ids(experiment_dir: Path) -> dict[str, set[str]]:
    """Collect cell IDs from checkpoints, raw results, and latest successful results."""
    checkpoint_store = CheckpointStore(experiment_dir / "checkpoints")
    checkpoint_completed = checkpoint_store.load_completed_cell_ids()
    checkpoint_failed = checkpoint_store.load_failed_cell_ids()

    records = load_results_records(experiment_dir / "results.jsonl")
    result_all = {record.cell_id for record in records}
    latest = latest_records_by_cell(records)
    result_completed = {
        cell_id
        for cell_id, record in latest.items()
        if record.status in {"completed", "budget_exhausted"}
    }
    result_failed = {
        cell_id for cell_id, record in latest.items() if record.status == "failed"
    }

    return {
        "checkpoint_completed": checkpoint_completed,
        "checkpoint_failed": checkpoint_failed,
        "result_all": result_all,
        "result_completed": result_completed,
        "result_failed": result_failed,
    }


def inspect_missing_cells(
    experiment_dir: Path,
    config: ExperimentConfig,
    *,
    config_dir: Path,
) -> dict[str, Any]:
    """Compare expected factorial cells against on-disk artifacts."""
    expected_specs = build_expected_cell_specs(config, config_dir=config_dir)
    expected_ids = {spec.cell_id for spec in expected_specs}
    expected_by_id = {spec.cell_id: spec for spec in expected_specs}

    observed = load_observed_cell_ids(experiment_dir)
    records = load_results_records(experiment_dir / "results.jsonl")
    duplicates = duplicate_cell_ids(records)

    missing_ids = sorted(expected_ids - observed["checkpoint_completed"])
    unexpected_ids = sorted(
        (observed["result_all"] | observed["checkpoint_completed"] | observed["checkpoint_failed"])
        - expected_ids
    )

    checkpoint_state: dict[str, Any] = {}
    state_path = experiment_dir / "checkpoint_state.json"
    if state_path.exists():
        checkpoint_state = json.loads(state_path.read_text(encoding="utf-8"))

    missing_cells = [expected_by_id[cell_id].to_dict() for cell_id in missing_ids]

    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "experiment_id": config.experiment_id,
        "experiment_dir": str(experiment_dir.resolve()),
        "config_dir": str(config_dir.resolve()),
        "counts": {
            "expected_cells": len(expected_ids),
            "checkpoint_completed_cells": len(observed["checkpoint_completed"]),
            "checkpoint_failed_cells": len(observed["checkpoint_failed"]),
            "result_cells": len(observed["result_all"]),
            "result_completed_cells": len(observed["result_completed"]),
            "result_failed_cells": len(observed["result_failed"]),
            "duplicate_cell_ids": len(duplicates),
            "missing_cell_ids": len(missing_ids),
            "unexpected_cell_ids": len(unexpected_ids),
        },
        "expected_cell_ids": sorted(expected_ids),
        "checkpoint_cell_ids": sorted(observed["checkpoint_completed"]),
        "checkpoint_failed_cell_ids": sorted(observed["checkpoint_failed"]),
        "result_cell_ids": sorted(observed["result_all"]),
        "result_completed_cell_ids": sorted(observed["result_completed"]),
        "result_failed_cell_ids": sorted(observed["result_failed"]),
        "duplicate_cell_ids": duplicates,
        "missing_cell_ids": missing_ids,
        "unexpected_cell_ids": unexpected_ids,
        "missing_cells": missing_cells,
        "checkpoint_state": checkpoint_state,
    }


def write_retry_spec(
    report: dict[str, Any],
    *,
    config_path: Path,
    output_path: Path,
) -> Path:
    """Write a retry specification containing only missing cells."""
    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "experiment_id": report["experiment_id"],
        "experiment_dir": report["experiment_dir"],
        "config_path": str(config_path.resolve()),
        "original_run_id": report.get("checkpoint_state", {}).get("run_id"),
        "missing_cells": report["missing_cells"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def render_missing_cells_markdown(report: dict[str, Any]) -> str:
    """Render a human-readable missing-cell diagnostic report."""
    counts = report["counts"]
    lines = [
        f"# Missing cells report: {report['experiment_id']}",
        "",
        f"_Generated at {report['generated_at']}_",
        "",
        "## Summary",
        "",
        f"- Expected cells: {counts['expected_cells']}",
        f"- Checkpoint cells (completed): {counts['checkpoint_completed_cells']}",
        f"- Checkpoint cells (failed): {counts['checkpoint_failed_cells']}",
        f"- Result cells (all): {counts['result_cells']}",
        f"- Result cells (completed): {counts['result_completed_cells']}",
        f"- Result cells (failed): {counts['result_failed_cells']}",
        f"- Duplicate cell IDs: {counts['duplicate_cell_ids']}",
        f"- Missing cell IDs: {counts['missing_cell_ids']}",
        f"- Unexpected cell IDs: {counts['unexpected_cell_ids']}",
        "",
    ]

    if report["duplicate_cell_ids"]:
        lines.extend(["## Duplicate cell IDs", ""])
        for cell_id in report["duplicate_cell_ids"]:
            lines.append(f"- `{cell_id}`")
        lines.append("")

    if report["unexpected_cell_ids"]:
        lines.extend(["## Unexpected cell IDs", ""])
        for cell_id in report["unexpected_cell_ids"]:
            lines.append(f"- `{cell_id}`")
        lines.append("")

    lines.extend(["## Missing cells", ""])
    if not report["missing_cells"]:
        lines.append("_No missing cells._")
    else:
        for cell in report["missing_cells"]:
            lines.extend(
                [
                    f"### `{cell['cell_id']}`",
                    "",
                    f"- model_id: `{cell['model_id']}`",
                    f"- provider: `{cell['provider']}`",
                    f"- task_id: `{cell['task_id']}`",
                    f"- prompt_variant_id: `{cell['prompt_variant_id']}`",
                    f"- temperature: {cell['temperature']}",
                    f"- run_index: {cell['run_index']}",
                    f"- seed: {cell['seed']}",
                    f"- rendered_prompt_hash: `{cell['rendered_prompt_hash']}`",
                    "",
                ]
            )

    return "\n".join(lines)


def write_missing_cells_report(
    experiment_dir: Path,
    report: dict[str, Any],
    *,
    write_retry_config: bool = False,
    config_path: Path | None = None,
) -> dict[str, Path]:
    """Write JSON/Markdown reports and optional retry spec."""
    if config_path is not None:
        report = {
            **report,
            "config_path": str(config_path.resolve()),
        }

    json_path = experiment_dir / "missing_cells_report.json"
    md_path = experiment_dir / "missing_cells_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_missing_cells_markdown(report), encoding="utf-8")

    outputs = {"json": json_path, "markdown": md_path}
    if write_retry_config:
        if config_path is None:
            msg = "config_path is required when write_retry_config=True"
            raise ValueError(msg)
        retry_path = experiment_dir / "retry_missing_cells.json"
        write_retry_spec(report, config_path=config_path, output_path=retry_path)
        outputs["retry_spec"] = retry_path
    return outputs


def combination_from_spec(
    config: ExperimentConfig,
    spec: dict[str, Any],
) -> ExperimentCombination:
    """Rebuild an ExperimentCombination from a retry spec entry."""
    return ExperimentCombination(
        run_index=int(spec["run_index"]),
        model_id=str(spec["model_id"]),
        provider=str(spec["provider"]),
        task_id=str(spec["task_id"]),
        prompt_variant_id=str(spec["prompt_variant_id"]),
        temperature=float(spec["temperature"]),
    )
