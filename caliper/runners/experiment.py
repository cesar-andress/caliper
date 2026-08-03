"""Orchestrate config-driven factorial experiment runs."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from caliper.config.schema import ExperimentConfig, PromptVariantConfig
from caliper.runners.cells import cell_to_dict, expand_cells
from caliper.runners.checkpoint import CheckpointStore
from caliper.runners.failures import (
    FailureWriter,
    count_terminal_completions,
    count_terminal_failures,
)
from caliper.runners.executor import (
    SUPPORTED_PROVIDER_TYPES,
    build_provider,
    build_task,
    execute_cell_safe,
)
from caliper.runners.pipeline import finalize_experiment, ensure_output_layout
from caliper.runners.progress import ExecutionProgress
from caliper.runners.results import ResultWriter
from caliper.utils.logging import setup_logging

logger = structlog.get_logger(__name__)

DEFAULT_OUTPUT_ROOT = Path("./outputs")
EXPERIMENTS_ROOT = Path("experiments")


@dataclass
class RunManifest:
    """Metadata describing a single experiment run."""

    run_id: str
    experiment_id: str
    started_at: datetime
    config_path: Path | None = None
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    status: str = "pending"
    finished_at: datetime | None = None
    total_cells: int = 0
    completed_cells: int = 0
    failed_cells: int = 0
    skipped_cells: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "started_at": self.started_at.isoformat(),
            "config_path": str(self.config_path) if self.config_path else None,
            "output_dir": str(self.output_dir),
            "status": self.status,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "total_cells": self.total_cells,
            "completed_cells": self.completed_cells,
            "failed_cells": self.failed_cells,
            "skipped_cells": self.skipped_cells,
            "metadata": self.metadata,
        }


def resolve_experiment_output_dir(
    config: ExperimentConfig,
    *,
    resume_dir: Path | None = None,
) -> Path:
    """Resolve the flat experiment output directory."""
    if resume_dir is not None:
        return resume_dir

    root = config.output.directory
    if root == DEFAULT_OUTPUT_ROOT or root.resolve() == (Path.cwd() / "outputs").resolve():
        root = EXPERIMENTS_ROOT
    return root / config.experiment_id


def detect_resume_dir(config: ExperimentConfig) -> Path | None:
    """Auto-detect an incomplete experiment directory for resume."""
    candidate = resolve_experiment_output_dir(config)
    jsonl_path = candidate / "results.jsonl"
    manifest_path = candidate / "manifest.json"
    if not candidate.exists():
        return None
    if jsonl_path.exists():
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                if data.get("status") == "completed":
                    return None
            except json.JSONDecodeError:
                pass
        return candidate
    checkpoint_dir = candidate / "checkpoints"
    if checkpoint_dir.exists() and any(checkpoint_dir.glob("*.json")):
        return candidate
    return None


class ExperimentRunner:
    """Execute the full factorial design for an experiment config."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        config_path: Path | None = None,
        dry_run: bool = False,
        resume_dir: Path | None = None,
        auto_resume: bool = True,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.config_dir = config_path.parent.resolve() if config_path else Path.cwd()
        self.dry_run = dry_run
        self.auto_resume = auto_resume

        if resume_dir is None and auto_resume and not dry_run:
            resume_dir = detect_resume_dir(config)

        self.resume_dir = resume_dir
        self.output_dir = resolve_experiment_output_dir(config, resume_dir=resume_dir)

        if resume_dir is not None:
            self.run_id = self._load_or_create_run_id(resume_dir)
        else:
            self.run_id = uuid.uuid4().hex[:12]

        self.manifest = RunManifest(
            run_id=self.run_id,
            experiment_id=config.experiment_id,
            started_at=datetime.now(tz=UTC),
            output_dir=self.output_dir,
            config_path=config_path,
        )
        self._execution_started_at = time.monotonic()

    def _load_or_create_run_id(self, resume_dir: Path) -> str:
        manifest_path = resume_dir / "manifest.json"
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                run_id = data.get("run_id")
                if isinstance(run_id, str) and run_id:
                    return run_id
            except json.JSONDecodeError:
                pass
        state_path = resume_dir / "checkpoint_state.json"
        if state_path.exists():
            try:
                data = json.loads(state_path.read_text(encoding="utf-8"))
                run_id = data.get("run_id")
                if isinstance(run_id, str) and run_id:
                    return run_id
            except json.JSONDecodeError:
                pass
        return resume_dir.name if resume_dir.name != self.config.experiment_id else uuid.uuid4().hex[:12]

    def setup(self) -> None:
        layout = ensure_output_layout(self.output_dir) if not self.dry_run else {}
        log_dir = layout.get("logs") if layout else None
        setup_logging(
            level=self.config.logging.level,
            log_format=self.config.logging.log_format,
            log_dir=log_dir if self.config.logging.log_to_file else None,
        )
        if not self.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        cells = expand_cells(self.config)
        logger.info(
            "experiment.start",
            run_id=self.run_id,
            experiment_id=self.config.experiment_id,
            dry_run=self.dry_run,
            output_dir=str(self.output_dir),
            auto_resume=self.resume_dir is not None,
            total_cells=len(cells),
            factorial_axes=self.config.factorial_axes(),
            supported_providers=sorted(SUPPORTED_PROVIDER_TYPES),
        )

    def run(self) -> RunManifest:
        self.setup()
        self.manifest.status = "running"
        cells = expand_cells(self.config)
        self.manifest.total_cells = len(cells)

        try:
            if self.dry_run:
                logger.info("experiment.dry_run", message="Skipping cell execution")
            else:
                self._execute(cells)
            self.manifest.status = "completed"
        except Exception:
            self.manifest.status = "failed"
            logger.exception("experiment.failed", run_id=self.run_id)
            raise
        finally:
            self.manifest.finished_at = datetime.now(tz=UTC)
            duration = time.monotonic() - self._execution_started_at
            if not self.dry_run:
                jsonl_path = self.output_dir / "results.jsonl"
                finalize_completed = self.manifest.completed_cells
                finalize_failed = self.manifest.failed_cells
                if jsonl_path.exists():
                    finalize_completed = count_terminal_completions(jsonl_path)
                    finalize_failed = count_terminal_failures(jsonl_path)
                manifest = finalize_experiment(
                    config=self.config,
                    config_path=self.config_path,
                    output_dir=self.output_dir,
                    run_id=self.run_id,
                    started_at=self.manifest.started_at,
                    finished_at=self.manifest.finished_at,
                    total_cells=self.manifest.total_cells,
                    completed_cells=finalize_completed,
                    failed_cells=finalize_failed,
                    skipped_cells=self.manifest.skipped_cells,
                    status=self.manifest.status,
                    execution_duration_seconds=duration,
                    result_paths={
                    k: Path(v) for k, v in self.manifest.metadata.get("result_paths", {}).items()
                }
                or None,
                )
                self.manifest.metadata["pipeline_manifest"] = manifest
            logger.info(
                "experiment.finish",
                run_id=self.run_id,
                status=self.manifest.status,
                completed=self.manifest.completed_cells,
                failed=self.manifest.failed_cells,
                skipped=self.manifest.skipped_cells,
            )

        return self.manifest

    def _execute(self, cells: list) -> None:
        layout = ensure_output_layout(self.output_dir)
        writer = ResultWriter(self.output_dir, self.config.experiment_id, self.run_id)
        checkpoint_store = CheckpointStore(layout["checkpoints"])
        failure_writer = FailureWriter(self.output_dir)

        completed_ids = writer.load_existing()
        checkpoint_ids = checkpoint_store.load_completed_cell_ids()
        completed_ids |= checkpoint_ids
        if completed_ids:
            logger.info("experiment.resume", completed_cells=len(completed_ids))

        checkpoint_store.save_state(
            {
                "run_id": self.run_id,
                "experiment_id": self.config.experiment_id,
                "updated_at": datetime.now(tz=UTC).isoformat(),
                "completed_cells": len(completed_ids),
            }
        )

        providers = {model.id: build_provider(self.config, model) for model in self.config.models}
        tasks = {task.id: build_task(self.config, task, self.config_dir) for task in self.config.tasks}
        prompts = self._resolve_prompts()

        progress = ExecutionProgress(total_cells=len(cells))
        total = len(cells)

        for index, cell in enumerate(cells, start=1):
            cell_info = cell_to_dict(self.config, cell)
            cell_id = str(cell_info["cell_id"])

            if writer.is_completed(cell_id) or cell_id in completed_ids:
                progress.record_skip()
                self.manifest.skipped_cells += 1
                logger.info(
                    "cell.skip",
                    progress=f"{index}/{total}",
                    reason="already_completed",
                    execution_progress=progress.to_dict(),
                    **cell_info,
                )
                continue

            logger.info(
                "cell.start",
                progress=f"{index}/{total}",
                execution_progress=progress.to_dict(),
                **cell_info,
            )

            record = execute_cell_safe(
                config=self.config,
                cell=cell,
                run_id=self.run_id,
                config_dir=self.config_dir,
                providers=providers,
                tasks=tasks,
                prompts=prompts,
            )
            writer.append(record)
            if record.status == "completed":
                checkpoint_store.write(record)
                completed_ids.add(cell_id)
            else:
                checkpoint_store.write_failed(record)
                failure_writer.append(record)

            progress.record_completion(success=record.status == "completed")
            if record.status == "completed":
                self.manifest.completed_cells += 1
            else:
                self.manifest.failed_cells += 1

            checkpoint_store.save_state(
                {
                    "run_id": self.run_id,
                    "experiment_id": self.config.experiment_id,
                    "updated_at": datetime.now(tz=UTC).isoformat(),
                    "completed_cells": count_terminal_completions(writer.jsonl_path),
                    "failed_cells": count_terminal_failures(writer.jsonl_path),
                    "execution_progress": progress.to_dict(),
                }
            )

            logger.info(
                "cell.finish",
                progress=f"{index}/{total}",
                status=record.status,
                score=record.score,
                cell_id=record.cell_id,
                execution_progress=progress.to_dict(),
            )

        result_paths = writer.finalize()
        self.manifest.metadata["result_paths"] = {k: str(v) for k, v in result_paths.items()}

    def _resolve_prompts(self) -> dict[str, PromptVariantConfig]:
        prompt_variants = self.config.prompt_variants or [
            PromptVariantConfig(id="default", template="{input}")
        ]
        prompts: dict[str, PromptVariantConfig] = {}
        for prompt in prompt_variants:
            if prompt.path is not None and not prompt.path.is_absolute():
                prompts[prompt.id] = prompt.model_copy(
                    update={"path": self.config_dir / prompt.path}
                )
            else:
                prompts[prompt.id] = prompt
        return prompts

    def plan_combinations(self) -> list[dict[str, Any]]:
        return [cell_to_dict(self.config, cell) for cell in expand_cells(self.config)]
