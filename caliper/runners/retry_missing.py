"""Retry only missing factorial cells with an auditable recovery workflow."""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from caliper.config.loader import load_config
from caliper.config.schema import PromptVariantConfig
from caliper.runners.checkpoint import CheckpointStore
from caliper.runners.executor import build_provider, build_task, execute_cell_safe
from caliper.runners.failures import count_terminal_completions, count_terminal_failures
from caliper.runners.missing_cells import combination_from_spec, inspect_missing_cells
from caliper.runners.pipeline import ensure_output_layout, finalize_experiment
from caliper.runners.results import ResultWriter

logger = structlog.get_logger(__name__)


def load_retry_spec(report_path: Path) -> dict[str, Any]:
    """Load a retry specification from inspect output or explicit retry spec."""
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if "missing_cells" not in payload:
        msg = f"Retry spec missing 'missing_cells' entries: {report_path}"
        raise ValueError(msg)
    return payload


def resolve_config_path(
    experiment_dir: Path,
    report: dict[str, Any],
    *,
    report_path: Path,
    config_path: Path | None = None,
) -> Path:
    """Resolve the frozen experiment config path for recovery."""
    if config_path is not None:
        return config_path.resolve()

    report_config = report.get("config_path")
    if isinstance(report_config, str) and report_config:
        return Path(report_config).resolve()

    retry_spec_path = experiment_dir / "retry_missing_cells.json"
    if retry_spec_path.exists():
        retry_payload = json.loads(retry_spec_path.read_text(encoding="utf-8"))
        retry_config = retry_payload.get("config_path")
        if isinstance(retry_config, str) and retry_config:
            return Path(retry_config).resolve()

    manifest_path = experiment_dir / "manifest.json"
    if manifest_path.exists():
        manifest_config = json.loads(manifest_path.read_text(encoding="utf-8")).get("config_path")
        if isinstance(manifest_config, str) and manifest_config:
            candidate = Path(manifest_config)
            if candidate.exists():
                return candidate.resolve()

    snapshot_path = experiment_dir / "config.yaml"
    if snapshot_path.exists():
        return snapshot_path.resolve()

    msg = (
        "Could not resolve experiment config. Pass --config/-c or regenerate "
        f"the report with inspect-missing-cells --write-retry-config ({report_path})."
    )
    raise ValueError(msg)


def _resolve_prompts(
    config,
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


def retry_missing_cells(
    experiment_dir: Path,
    *,
    report_path: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Execute only missing cells and regenerate final artifacts."""
    report = load_retry_spec(report_path)
    config_file = resolve_config_path(
        experiment_dir,
        report,
        report_path=report_path,
        config_path=config_path,
    )
    config = load_config(config_file)
    config_dir = config_file.parent.resolve()

    report_experiment_dir = report.get("experiment_dir")
    if report_experiment_dir is not None and str(experiment_dir.resolve()) != str(
        Path(report_experiment_dir).resolve()
    ):
        msg = (
            "experiment_dir does not match report.experiment_dir: "
            f"{experiment_dir} != {report['experiment_dir']}"
        )
        raise ValueError(msg)

    original_run_id = report.get("original_run_id")
    if not original_run_id:
        manifest_path = experiment_dir / "manifest.json"
        if manifest_path.exists():
            original_run_id = json.loads(manifest_path.read_text(encoding="utf-8")).get("run_id")
    if not original_run_id:
        state_path = experiment_dir / "checkpoint_state.json"
        if state_path.exists():
            original_run_id = json.loads(state_path.read_text(encoding="utf-8")).get("run_id")
    if not original_run_id:
        original_run_id = experiment_dir.name

    recovery_run_id = uuid.uuid4().hex[:12]
    started_at = datetime.now(tz=UTC)
    execution_started = time.monotonic()

    layout = ensure_output_layout(experiment_dir)
    writer = ResultWriter(experiment_dir, config.experiment_id, original_run_id)
    checkpoint_store = CheckpointStore(layout["checkpoints"])
    completed_ids = writer.load_existing() | checkpoint_store.load_completed_cell_ids()

    providers = {model.id: build_provider(config, model) for model in config.models}
    tasks = {task.id: build_task(config, task, config_dir) for task in config.tasks}
    prompts = _resolve_prompts(config, config_dir)

    audit_path = experiment_dir / "recovery_audit.jsonl"
    recovered = 0
    skipped = 0
    still_failed = 0

    for spec in report["missing_cells"]:
        cell = combination_from_spec(config, spec)
        cell_id = str(spec["cell_id"])

        if cell_id in completed_ids:
            skipped += 1
            audit_entry = {
                "recovery_run_id": recovery_run_id,
                "original_run_id": original_run_id,
                "cell_id": cell_id,
                "action": "skip_already_completed",
                "timestamp": datetime.now(tz=UTC).isoformat(),
            }
            with audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(audit_entry))
                handle.write("\n")
            continue

        record = execute_cell_safe(
            config=config,
            cell=cell,
            run_id=original_run_id,
            config_dir=config_dir,
            providers=providers,
            tasks=tasks,
            prompts=prompts,
            output_dir=experiment_dir,
        )
        record.metadata = {
            **record.metadata,
            "recovery_run_id": recovery_run_id,
            "recovered_from_missing_cells_report": str(report_path.resolve()),
        }
        writer.append(record)

        if record.status in {"completed", "budget_exhausted"}:
            checkpoint_store.write(record)
            completed_ids.add(cell_id)
            recovered += 1
            action = "recovered"
        else:
            checkpoint_store.write_failed(record)
            from caliper.runners.failures import FailureWriter

            FailureWriter(experiment_dir).append(record)
            still_failed += 1
            action = "retry_failed"

        audit_entry = {
            "recovery_run_id": recovery_run_id,
            "original_run_id": original_run_id,
            "cell_id": cell_id,
            "action": action,
            "status": record.status,
            "error": record.error,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(audit_entry))
            handle.write("\n")

    result_paths = writer.finalize()
    finished_at = datetime.now(tz=UTC)
    duration = time.monotonic() - execution_started

    jsonl_path = experiment_dir / "results.jsonl"
    completed_total = count_terminal_completions(jsonl_path)
    failed_total = count_terminal_failures(jsonl_path)
    total_cells = config.total_combinations()

    manifest = finalize_experiment(
        config=config,
        config_path=config_file,
        output_dir=experiment_dir,
        run_id=original_run_id,
        started_at=started_at,
        finished_at=finished_at,
        total_cells=total_cells,
        completed_cells=completed_total,
        failed_cells=failed_total,
        skipped_cells=max(total_cells - completed_total - failed_total, 0),
        status="completed",
        execution_duration_seconds=duration,
        result_paths=result_paths,
    )
    manifest["original_run_id"] = original_run_id
    manifest["recovery_run_id"] = recovery_run_id
    manifest["recovery"] = {
        "report_path": str(report_path.resolve()),
        "recovered_cells": recovered,
        "skipped_cells": skipped,
        "still_failed_cells": still_failed,
        "audit_path": str(audit_path.resolve()),
    }

    manifest_path = experiment_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    checkpoint_store.save_state(
        {
            "run_id": original_run_id,
            "experiment_id": config.experiment_id,
            "updated_at": finished_at.isoformat(),
            "completed_cells": completed_total,
            "failed_cells": failed_total,
            "recovery_run_id": recovery_run_id,
        }
    )

    post_report = inspect_missing_cells(experiment_dir, config, config_dir=config_dir)
    return {
        "original_run_id": original_run_id,
        "recovery_run_id": recovery_run_id,
        "recovered_cells": recovered,
        "skipped_cells": skipped,
        "still_failed_cells": still_failed,
        "remaining_missing_cells": post_report["counts"]["missing_cell_ids"],
        "manifest_path": str(manifest_path),
        "audit_path": str(audit_path),
    }
