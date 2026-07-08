"""Post-hoc evaluation of experiment result files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from caliper.config.schema import ExperimentConfig, TaskConfig
from caliper.evaluation.context import EvaluationInput
from caliper.evaluation.registry import EvaluationOptions, evaluate_sample
from caliper.evaluation.schema import CellEvaluationRecord
from caliper.runners.executor import resolve_dataset_path, resolve_task_domain
from caliper.storage.formats import read_results, write_results
from caliper.tasks import create_task

logger = structlog.get_logger(__name__)


def _reference_for_task(
    task_cfg: TaskConfig, config_dir: Path
) -> tuple[str | None, list[str] | None, str, dict[str, Any]]:
    """Load the first benchmark instance reference for a configured task."""
    domain = resolve_task_domain(task_cfg)
    dataset_path = resolve_dataset_path(task_cfg, config_dir)
    task_config: dict[str, Any] = {}
    if task_cfg.num_samples is not None:
        task_config["num_samples"] = task_cfg.num_samples
    task_config.update(task_cfg.extra)

    task = create_task(domain, task_cfg.id, dataset_path, **task_config)
    examples = task.load_examples()
    if not examples:
        return None, None, "python", {}
    example = examples[0]
    metadata = {**example.extra, "prompt_stub": example.input}
    return example.expected_output, example.tests, example.language, metadata


def evaluate_results_dataframe(
    df: pd.DataFrame,
    config: ExperimentConfig,
    *,
    config_dir: Path,
    options: EvaluationOptions | None = None,
) -> list[CellEvaluationRecord]:
    """Evaluate every completed row in a results DataFrame."""
    if df.empty:
        return []

    task_cfg_by_id = {task.id: task for task in config.tasks}
    reference_cache: dict[str, tuple[str | None, list[str] | None, str, dict[str, Any]]] = {}
    records: list[CellEvaluationRecord] = []

    for row in df.to_dict(orient="records"):
        if row.get("status") != "completed":
            logger.info(
                "evaluate.skip_row",
                cell_id=row.get("cell_id"),
                reason="status_not_completed",
            )
            continue

        task_id = str(row["task_id"])
        task_cfg = task_cfg_by_id.get(task_id)
        if task_cfg is None:
            logger.warning("evaluate.skip_row", cell_id=row.get("cell_id"), reason="unknown_task")
            continue

        if task_id not in reference_cache:
            reference_cache[task_id] = _reference_for_task(task_cfg, config_dir)

        expected_output, tests, language, task_metadata = reference_cache[task_id]
        domain = resolve_task_domain(task_cfg)
        prediction = str(row.get("prediction", ""))

        sample = EvaluationInput(
            prediction=prediction,
            expected_output=expected_output,
            tests=tests,
            domain=domain,
            language=language,
            metadata={"task_id": task_id, **task_metadata},
        )
        metrics = evaluate_sample(
            sample,
            options=options,
            metric_names=list(task_cfg.metrics or config.evaluation_metrics),
        )

        records.append(
            CellEvaluationRecord(
                cell_id=str(row["cell_id"]),
                experiment_id=str(row.get("experiment_id", config.experiment_id)),
                run_id=str(row.get("run_id", "")),
                task_id=task_id,
                domain=domain,
                model_id=str(row.get("model_id", "")),
                prediction=prediction,
                expected_output=expected_output,
                metrics=metrics,
                metadata={
                    "run_index": row.get("run_index"),
                    "prompt_variant_id": row.get("prompt_variant_id"),
                    "temperature": row.get("temperature"),
                },
            )
        )

    return records


def evaluate_results_file(
    results_path: Path,
    config: ExperimentConfig,
    *,
    config_path: Path | None = None,
    options: EvaluationOptions | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate a saved results file and write evaluation outputs."""
    config_dir = config_path.parent.resolve() if config_path else Path.cwd()
    df = read_results(results_path)

    records = evaluate_results_dataframe(
        df,
        config,
        config_dir=config_dir,
        options=options,
    )

    if output_dir is None:
        output_dir = results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "evaluations.jsonl"
    parquet_path = output_dir / "evaluations.parquet"

    flat_rows: list[dict[str, Any]] = []
    for record in records:
        base = {
            "cell_id": record.cell_id,
            "experiment_id": record.experiment_id,
            "run_id": record.run_id,
            "task_id": record.task_id,
            "domain": record.domain,
            "model_id": record.model_id,
            "prediction": record.prediction,
            "expected_output": record.expected_output,
            "evaluated_at": record.evaluated_at,
        }
        for metric in record.metrics:
            flat_rows.append(
                {
                    **base,
                    "metric_name": metric.name,
                    "metric_value": metric.value,
                    "metric_success": metric.success,
                    "metric_metadata": metric.metadata,
                }
            )

    if flat_rows:
        eval_df = pd.DataFrame(flat_rows)
        write_results(eval_df, parquet_path, fmt="parquet")
        eval_df.to_json(jsonl_path, orient="records", lines=True)

    summary = {
        "rows_evaluated": len(records),
        "metrics_per_row": len(records[0].metrics) if records else 0,
        "output_jsonl": str(jsonl_path) if flat_rows else None,
        "output_parquet": str(parquet_path) if flat_rows else None,
    }
    logger.info("evaluate.complete", **summary)
    return summary
