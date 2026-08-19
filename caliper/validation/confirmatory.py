"""End-to-end pre-flight validation for confirmatory experiments."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from caliper.benchmarks.prompts import CONTROLLED_OUTPUT_SUFFIX, controlled_prompt_templates
from caliper.config.loader import validate_config
from caliper.config.metrics import resolve_primary_metric
from caliper.evaluation.code_extraction import extract_python_code
from caliper.evaluation.sandbox import ExecutionLimits, execute_python_program
from caliper.runners.checkpoint import CheckpointStore
from caliper.runners.experiment import ExperimentRunner
from caliper.runners.executor import build_provider, build_task
from caliper.runners.reproducibility import collect_environment
from caliper.runners.results import ExperimentResultRecord
from caliper.statistics.robustness_report import run_robustness_analysis
from caliper.storage.formats import read_results
from caliper.tasks.loader import TaskDataset
from caliper.validation.config_builder import (
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    PROMPT_PROTOCOL_VERSION,
    build_preflight_config,
    dataset_path,
    reference_config_path,
    resolve_benchmark,
    verify_prompt_family,
)
from caliper.validation.report import write_reports
from caliper.validation.types import (
    Severity,
    StageResult,
    StageStatus,
    TimingBreakdown,
    ValidationReport,
    ValidationStage,
)


def _stage(
    stage: ValidationStage,
    status: StageStatus,
    *,
    latency_ms: float = 0.0,
    message: str = "",
    root_cause: str | None = None,
    recommended_fix: str | None = None,
    severity: Severity | None = None,
    details: dict[str, Any] | None = None,
) -> StageResult:
    if status == "FAIL" and severity is None:
        severity = "critical"
    return StageResult(
        stage=stage,
        status=status,
        latency_ms=latency_ms,
        message=message,
        root_cause=root_cause,
        recommended_fix=recommended_fix,
        severity=severity,
        details=details or {},
    )


def _file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu_summary(env: dict[str, Any]) -> str:
    gpu = env.get("gpu", {})
    if gpu.get("gpu_available") or gpu.get("device_name"):
        name = gpu.get("device_name") or gpu.get("name") or "GPU detected"
        mem = gpu.get("memory_total_gb")
        if mem:
            return f"{name} ({mem} GB)"
        return str(name)
    return "No GPU detected (CPU-only execution)"


def _disk_free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def validate_benchmark_load(
    benchmark: str,
    *,
    num_tasks: int,
    expected_total_tasks: int | None = None,
) -> tuple[StageResult, dict[str, Any]]:
    start = time.perf_counter()
    info: dict[str, Any] = {"name": resolve_benchmark(benchmark)}
    try:
        resolved = resolve_benchmark(benchmark)
        path = dataset_path(benchmark)
        if not path.exists():
            return (
                _stage(
                    ValidationStage.BENCHMARK_LOAD,
                    "FAIL",
                    latency_ms=(time.perf_counter() - start) * 1000,
                    message=f"Dataset missing: {path}",
                    root_cause="Benchmark JSONL not materialized",
                    recommended_fix="Run: caliper benchmarks materialize --benchmark "
                    f"{benchmark}",
                    severity="critical",
                ),
                info,
            )

        dataset = TaskDataset.from_jsonl(path)
        if expected_total_tasks is not None and len(dataset) != expected_total_tasks:
            return (
                _stage(
                    ValidationStage.BENCHMARK_LOAD,
                    "FAIL",
                    latency_ms=(time.perf_counter() - start) * 1000,
                    message=(
                        f"Dataset has {len(dataset)} tasks, expected exactly {expected_total_tasks}"
                    ),
                    root_cause="Benchmark task count mismatch",
                    recommended_fix="Re-materialize HumanEval+ dataset",
                    severity="critical",
                ),
                info,
            )
        if len(dataset) < num_tasks:
            return (
                _stage(
                    ValidationStage.BENCHMARK_LOAD,
                    "FAIL",
                    latency_ms=(time.perf_counter() - start) * 1000,
                    message=f"Dataset has {len(dataset)} tasks, need {num_tasks}",
                    root_cause="Insufficient benchmark tasks",
                    recommended_fix="Re-materialize benchmark or reduce --tasks",
                    severity="critical",
                ),
                info,
            )

        manifest_path = path.parent / f"{resolved}_manifest.json"
        version = "unknown"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            version = manifest.get("version", version)

        info.update(
            {
                "dataset_path": str(path.resolve()),
                "checksum": _file_checksum(path),
                "version": version,
                "num_tasks": len(dataset),
                "tasks_exercised": num_tasks,
            }
        )
        return (
            _stage(
                ValidationStage.BENCHMARK_LOAD,
                "PASS",
                latency_ms=(time.perf_counter() - start) * 1000,
                message=f"Loaded {len(dataset)} tasks from {path.name}",
            ),
            info,
        )
    except Exception as exc:  # noqa: BLE001
        return (
            _stage(
                ValidationStage.BENCHMARK_LOAD,
                "FAIL",
                latency_ms=(time.perf_counter() - start) * 1000,
                message=str(exc),
                root_cause="Benchmark loader error",
                recommended_fix="Verify benchmark JSONL integrity and schema",
                severity="critical",
            ),
            info,
        )


def validate_task_metadata(benchmark: str, task_ids: list[str]) -> StageResult:
    start = time.perf_counter()
    try:
        path = dataset_path(benchmark)
        dataset = TaskDataset.from_jsonl(path)
        id_set = set(task_ids)
        selected = [r for r in dataset.records if r.task_id in id_set]
        if len(selected) != len(task_ids):
            missing = id_set - {r.task_id for r in selected}
            return _stage(
                ValidationStage.TASK_METADATA,
                "FAIL",
                latency_ms=(time.perf_counter() - start) * 1000,
                message=f"Missing task ids: {sorted(missing)}",
                root_cause="Task subset not found in dataset",
                recommended_fix="Regenerate pre-flight config task selection",
                severity="critical",
            )

        required = ("input", "tests", "language", "source_benchmark")
        for record in selected:
            if record.domain != "executable_code_generation":
                return _stage(
                    ValidationStage.TASK_METADATA,
                    "FAIL",
                    latency_ms=(time.perf_counter() - start) * 1000,
                    message=f"Task {record.task_id} has domain {record.domain}",
                    root_cause="Wrong task domain for confirmatory study",
                    recommended_fix="Re-materialize benchmark with executable_code_generation domain",
                    severity="high",
                )
            for field in required:
                if field == "tests" and not record.tests:
                    return _stage(
                        ValidationStage.TASK_METADATA,
                        "FAIL",
                        latency_ms=(time.perf_counter() - start) * 1000,
                        message=f"Task {record.task_id} has no tests",
                        root_cause="Missing unit tests",
                        recommended_fix="Re-materialize benchmark dataset",
                        severity="critical",
                    )
                if field != "tests" and not getattr(record, field, None):
                    return _stage(
                        ValidationStage.TASK_METADATA,
                        "FAIL",
                        latency_ms=(time.perf_counter() - start) * 1000,
                        message=f"Task {record.task_id} missing {field}",
                        root_cause="Incomplete task metadata",
                        recommended_fix="Re-materialize benchmark dataset",
                        severity="high",
                    )

        return _stage(
            ValidationStage.TASK_METADATA,
            "PASS",
            latency_ms=(time.perf_counter() - start) * 1000,
            message=f"Validated metadata for {len(selected)} tasks",
            details={"task_ids": task_ids},
        )
    except Exception as exc:  # noqa: BLE001
        return _stage(
            ValidationStage.TASK_METADATA,
            "FAIL",
            latency_ms=(time.perf_counter() - start) * 1000,
            message=str(exc),
            root_cause="Task metadata validation error",
            recommended_fix="Inspect benchmark JSONL records",
            severity="critical",
        )


def validate_prompt_generation(prompt_id: str, task_input: str) -> StageResult:
    start = time.perf_counter()
    errors = verify_prompt_family()
    if errors:
        return _stage(
            ValidationStage.PROMPT_GENERATION,
            "FAIL",
            latency_ms=(time.perf_counter() - start) * 1000,
            message="; ".join(errors),
            root_cause="Prompt protocol violation",
            recommended_fix="Ensure all confirmatory prompts include CONTROLLED_OUTPUT_SUFFIX",
            severity="critical",
        )

    prompt = next((p for p in controlled_prompt_templates() if p.style == prompt_id), None)
    if prompt is None:
        return _stage(
            ValidationStage.PROMPT_GENERATION,
            "FAIL",
            latency_ms=(time.perf_counter() - start) * 1000,
            message=f"Unknown prompt '{prompt_id}'",
            root_cause="Invalid prompt variant",
            recommended_fix=f"Use one of: {[p.style for p in controlled_prompt_templates()]}",
            severity="high",
        )

    rendered = prompt.render(task_input)
    if CONTROLLED_OUTPUT_SUFFIX.strip() not in rendered:
        return _stage(
            ValidationStage.PROMPT_GENERATION,
            "FAIL",
            latency_ms=(time.perf_counter() - start) * 1000,
            message="Rendered prompt missing controlled output suffix",
            root_cause="Invalid prompt template",
            recommended_fix="Fix prompt template in confirmatory config",
            severity="critical",
        )

    return _stage(
        ValidationStage.PROMPT_GENERATION,
        "PASS",
        latency_ms=(time.perf_counter() - start) * 1000,
        message=f"Prompt '{prompt_id}' renders with controlled output format",
    )


def validate_model_provider(config: Any, model_id: str) -> StageResult:
    start = time.perf_counter()
    try:
        model = next(m for m in config.models if m.id == model_id)
        provider = build_provider(config, model)
        if not provider.is_available():
            return _stage(
                ValidationStage.MODEL_PROVIDER,
                "FAIL",
                latency_ms=(time.perf_counter() - start) * 1000,
                message=f"Provider for model '{model_id}' is unavailable",
                root_cause="Model provider unavailable",
                recommended_fix="Start Ollama and pull the required model",
                severity="critical",
            )
        return _stage(
            ValidationStage.MODEL_PROVIDER,
            "PASS",
            latency_ms=(time.perf_counter() - start) * 1000,
            message=f"Provider ready for model '{model_id}'",
        )
    except StopIteration:
        return _stage(
            ValidationStage.MODEL_PROVIDER,
            "FAIL",
            latency_ms=(time.perf_counter() - start) * 1000,
            message=f"Model '{model_id}' not in configuration",
            root_cause="Model missing from config",
            recommended_fix="Use a model listed in confirmatory YAML",
            severity="critical",
        )
    except Exception as exc:  # noqa: BLE001
        return _stage(
            ValidationStage.MODEL_PROVIDER,
            "FAIL",
            latency_ms=(time.perf_counter() - start) * 1000,
            message=str(exc),
            root_cause="Provider initialization failed",
            recommended_fix="Check Ollama configuration and provider settings",
            severity="critical",
        )


def validate_ollama_connectivity(config: Any, model_id: str) -> StageResult:
    start = time.perf_counter()
    try:
        from caliper.models.ollama_client import OllamaConnectionError
        from caliper.models.ollama_provider import list_local_models

        model = next(m for m in config.models if m.id == model_id)
        provider_cfg = config.providers[model.provider]
        base_url = provider_cfg.base_url or "http://localhost:11434"
        models = list_local_models(base_url=base_url)
        if model.model_id not in models:
            return _stage(
                ValidationStage.OLLAMA_CONNECTIVITY,
                "FAIL",
                latency_ms=(time.perf_counter() - start) * 1000,
                message=f"Model '{model.model_id}' not in Ollama at {base_url}",
                root_cause="Model missing from Ollama",
                recommended_fix=f"Run: ollama pull {model.model_id}",
                severity="critical",
                details={"available_models": models[:20]},
            )
        return _stage(
            ValidationStage.OLLAMA_CONNECTIVITY,
            "PASS",
            latency_ms=(time.perf_counter() - start) * 1000,
            message=f"Ollama reachable; model '{model.model_id}' available",
            details={"base_url": base_url},
        )
    except OllamaConnectionError as exc:
        return _stage(
            ValidationStage.OLLAMA_CONNECTIVITY,
            "FAIL",
            latency_ms=(time.perf_counter() - start) * 1000,
            message=str(exc),
            root_cause="Ollama unavailable",
            recommended_fix="Start Ollama: systemctl start ollama (or ollama serve)",
            severity="critical",
        )
    except Exception as exc:  # noqa: BLE001
        return _stage(
            ValidationStage.OLLAMA_CONNECTIVITY,
            "FAIL",
            latency_ms=(time.perf_counter() - start) * 1000,
            message=str(exc),
            root_cause="Ollama connectivity check failed",
            recommended_fix="Verify Ollama is running and base_url is correct",
            severity="critical",
        )


def validate_timeout_handling() -> StageResult:
    start = time.perf_counter()
    program = "while True:\n    pass\n"
    result = execute_python_program(program, limits=ExecutionLimits(timeout_seconds=0.2, memory_mb=128))
    if result.timed_out:
        return _stage(
            ValidationStage.TIMEOUT_HANDLING,
            "PASS",
            latency_ms=(time.perf_counter() - start) * 1000,
            message="Sandbox timeout enforced correctly",
        )
    return _stage(
        ValidationStage.TIMEOUT_HANDLING,
        "FAIL",
        latency_ms=(time.perf_counter() - start) * 1000,
        message="Sandbox did not timeout infinite loop",
        root_cause="Timeout not enforced",
        recommended_fix="Check subprocess timeout configuration in sandbox.py",
        severity="critical",
    )


def validate_resume_mechanism(output_dir: Path) -> StageResult:
    start = time.perf_counter()
    try:
        store = CheckpointStore(output_dir / "checkpoints")
        record = ExperimentResultRecord(
            cell_id="preflight-resume-test",
            experiment_id="preflight",
            run_id="preflight-run",
            run_index=0,
            model_id="m1",
            provider_name="ollama_local",
            provider_type="ollama",
            task_id="t1",
            prompt_variant_id="minimal",
            temperature=0.0,
            seed=42,
            metric="pass_at_1",
            score=1.0,
            scores={"pass_at_1": 1.0},
            prediction="def f(): pass",
            num_examples=1,
            latency_ms=1.0,
            status="completed",
        )
        store.write(record)
        loaded = store.load_completed_cell_ids()
        if "preflight-resume-test" not in loaded:
            return _stage(
                ValidationStage.RESUME_MECHANISM,
                "FAIL",
                latency_ms=(time.perf_counter() - start) * 1000,
                message="Checkpoint write/read failed",
                root_cause="Checkpoint store not persisting",
                recommended_fix="Inspect checkpoints directory permissions",
                severity="high",
            )
        return _stage(
            ValidationStage.RESUME_MECHANISM,
            "PASS",
            latency_ms=(time.perf_counter() - start) * 1000,
            message="Checkpoint write and reload succeeded",
        )
    except Exception as exc:  # noqa: BLE001
        return _stage(
            ValidationStage.RESUME_MECHANISM,
            "FAIL",
            latency_ms=(time.perf_counter() - start) * 1000,
            message=str(exc),
            root_cause="Resume mechanism error",
            recommended_fix="Verify checkpoint directory is writable",
            severity="high",
        )


def _validate_post_run_artifacts(output_dir: Path, config: Any) -> list[StageResult]:
    results: list[StageResult] = []

    parquet_path = output_dir / "results.parquet"
    start = time.perf_counter()
    if parquet_path.exists():
        df = read_results(parquet_path)
        results.append(
            _stage(
                ValidationStage.PARQUET_EXPORT,
                "PASS",
                latency_ms=(time.perf_counter() - start) * 1000,
                message=f"results.parquet written ({len(df)} rows)",
            )
        )
    else:
        results.append(
            _stage(
                ValidationStage.PARQUET_EXPORT,
                "FAIL",
                latency_ms=(time.perf_counter() - start) * 1000,
                message="results.parquet missing",
                root_cause="Experiment did not write parquet results",
                recommended_fix="Inspect experiment logs for cell failures",
                severity="critical",
            )
        )
        return results

    df = read_results(parquet_path)
    completed = df[df["status"] == "completed"]

    start = time.perf_counter()
    stats_path = output_dir / "statistical_dataset.parquet"
    if stats_path.exists():
        results.append(
            _stage(
                ValidationStage.STATISTICAL_DATASET,
                "PASS",
                latency_ms=(time.perf_counter() - start) * 1000,
                message="statistical_dataset.parquet generated",
            )
        )
    else:
        results.append(
            _stage(
                ValidationStage.STATISTICAL_DATASET,
                "FAIL",
                latency_ms=(time.perf_counter() - start) * 1000,
                message="statistical_dataset.parquet missing",
                root_cause="Statistics cannot run without completed results",
                recommended_fix="Ensure at least one cell completed successfully",
                severity="critical",
            )
        )

    start = time.perf_counter()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        results.append(
            _stage(
                ValidationStage.MANIFEST_GENERATION,
                "PASS",
                latency_ms=(time.perf_counter() - start) * 1000,
                message=f"manifest.json status={manifest.get('status')}",
            )
        )
    else:
        results.append(
            _stage(
                ValidationStage.MANIFEST_GENERATION,
                "FAIL",
                latency_ms=(time.perf_counter() - start) * 1000,
                message="manifest.json missing",
                root_cause="Pipeline finalization incomplete",
                recommended_fix="Check finalize_experiment logs",
                severity="critical",
            )
        )

    start = time.perf_counter()
    report_path = output_dir / "report.md"
    if report_path.exists() and report_path.stat().st_size > 0:
        results.append(
            _stage(
                ValidationStage.REPORT_GENERATION,
                "PASS",
                latency_ms=(time.perf_counter() - start) * 1000,
                message="report.md generated",
            )
        )
    else:
        results.append(
            _stage(
                ValidationStage.REPORT_GENERATION,
                "FAIL",
                latency_ms=(time.perf_counter() - start) * 1000,
                message="report.md missing or empty",
                root_cause="Report generation failed",
                recommended_fix="Inspect generate_report pipeline step",
                severity="high",
            )
        )

    start = time.perf_counter()
    artifact_complete = False
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifact = manifest.get("artifact", {})
        artifact_complete = bool(artifact.get("complete"))
        if artifact_complete:
            results.append(
                _stage(
                    ValidationStage.ARTIFACT_EXPORT,
                    "PASS",
                    latency_ms=(time.perf_counter() - start) * 1000,
                    message=f"Artifact bundle complete at {artifact.get('path')}",
                )
            )
        else:
            results.append(
                _stage(
                    ValidationStage.ARTIFACT_EXPORT,
                    "FAIL",
                    latency_ms=(time.perf_counter() - start) * 1000,
                    message="Artifact export incomplete",
                    root_cause="Artifact verification failed",
                    recommended_fix=f"Inspect artifact errors: {artifact.get('errors', [])}",
                    severity="high",
                    details=artifact,
                )
            )
    else:
        results.append(
            _stage(
                ValidationStage.ARTIFACT_EXPORT,
                "FAIL",
                latency_ms=(time.perf_counter() - start) * 1000,
                message="Cannot verify artifact without manifest",
                root_cause="Manifest missing",
                recommended_fix="Complete experiment pipeline first",
                severity="high",
            )
        )

    start = time.perf_counter()
    try:
        if stats_path.exists():
            stats_df = pd.read_parquet(stats_path)
            n_obs = len(stats_df)
            if n_obs < 10:
                results.append(
                    _stage(
                        ValidationStage.ROBUSTNESS_PIPELINE,
                        "WARN",
                        latency_ms=(time.perf_counter() - start) * 1000,
                        message=f"Robustness deferred — only {n_obs} observations in pre-flight (needs ≥10)",
                        severity="medium",
                        recommended_fix="Robustness will run on the full 9,600-cell confirmatory study",
                    )
                )
            else:
                run_robustness_analysis(output_dir, n_bootstrap=500)
                robust_dir = output_dir / "paper1_analysis" / "robustness"
                if (robust_dir / "robustness_manifest.json").exists():
                    results.append(
                        _stage(
                            ValidationStage.ROBUSTNESS_PIPELINE,
                            "PASS",
                            latency_ms=(time.perf_counter() - start) * 1000,
                            message="Robustness pipeline completed (fast bootstrap)",
                        )
                    )
                else:
                    results.append(
                        _stage(
                            ValidationStage.ROBUSTNESS_PIPELINE,
                            "FAIL",
                            latency_ms=(time.perf_counter() - start) * 1000,
                            message="Robustness outputs missing",
                            root_cause="Robustness pipeline did not produce outputs",
                            recommended_fix="Run caliper analyze robustness --fast manually",
                            severity="high",
                        )
                    )
        else:
            results.append(
                _stage(
                    ValidationStage.ROBUSTNESS_PIPELINE,
                    "SKIP",
                    latency_ms=(time.perf_counter() - start) * 1000,
                    message="Skipped — no statistical dataset",
                    severity="medium",
                )
            )
    except Exception as exc:  # noqa: BLE001
        results.append(
            _stage(
                ValidationStage.ROBUSTNESS_PIPELINE,
                "FAIL",
                latency_ms=(time.perf_counter() - start) * 1000,
                message=str(exc),
                root_cause="Robustness pipeline error",
                recommended_fix="Verify statistical_dataset.parquet schema matches Paper 1 expectations",
                severity="high",
            )
        )

    _ = config
    _ = completed
    return results


def _validate_inference_and_metrics(output_dir: Path) -> list[StageResult]:
    results: list[StageResult] = []
    parquet_path = output_dir / "results.parquet"
    if not parquet_path.exists():
        return [
            _stage(
                ValidationStage.INFERENCE,
                "FAIL",
                message="No results to inspect",
                root_cause="Experiment produced no parquet output",
                recommended_fix="Fix upstream experiment execution failures",
                severity="critical",
            )
        ]

    df = read_results(parquet_path)
    completed = df[df["status"] == "completed"]

    start = time.perf_counter()
    if completed.empty:
        results.append(
            _stage(
                ValidationStage.INFERENCE,
                "FAIL",
                latency_ms=(time.perf_counter() - start) * 1000,
                message="No completed cells",
                root_cause="All inference cells failed",
                recommended_fix="Check Ollama logs and model availability",
                severity="critical",
            )
        )
        return results

    empty_preds = completed["prediction"].astype(str).str.strip().eq("")
    budget_mask = completed["status"].astype(str).eq("budget_exhausted") if "status" in completed.columns else False
    n_budget = int(budget_mask.sum()) if hasattr(budget_mask, "sum") else 0
    n_empty = int(empty_preds.sum())
    if n_budget or n_empty:
        results.append(
            _stage(
                ValidationStage.INFERENCE,
                "FAIL",
                latency_ms=(time.perf_counter() - start) * 1000,
                message=(
                    f"{n_empty} empty predictions "
                    f"({n_budget} marked budget_exhausted)"
                ),
                root_cause=(
                    "Empty visible response; for reasoning models this often means "
                    "thinking consumed num_predict (done_reason=length)"
                ),
                recommended_fix=(
                    "For Qwen3-class Ollama models: set think=false and/or raise "
                    "max_tokens; inspect done_reason/eval_count/thinking_length"
                ),
                severity="critical",
            )
        )
    else:
        results.append(
            _stage(
                ValidationStage.INFERENCE,
                "PASS",
                latency_ms=(time.perf_counter() - start) * 1000,
                message=f"{len(completed)} successful inferences",
            )
        )

    start = time.perf_counter()
    extracted = 0
    for pred in completed["prediction"].astype(str):
        if extract_python_code(pred).strip():
            extracted += 1
    if extracted == 0:
        results.append(
            _stage(
                ValidationStage.CODE_EXTRACTION,
                "FAIL",
                latency_ms=(time.perf_counter() - start) * 1000,
                message="No extractable Python code in predictions",
                root_cause="Code extraction failed on all predictions",
                recommended_fix="Verify models follow ```python fenced output format",
                severity="high",
            )
        )
    else:
        results.append(
            _stage(
                ValidationStage.CODE_EXTRACTION,
                "PASS",
                latency_ms=(time.perf_counter() - start) * 1000,
                message=f"Extracted code from {extracted}/{len(completed)} predictions",
            )
        )

    start = time.perf_counter()
    has_pass = "scores" in completed.columns and completed["scores"].notna().any()
    pass_values: list[float] = []
    sandbox_latencies: list[float] = []
    syntax_values: list[float] = []
    if has_pass:
        for raw_scores in completed["scores"]:
            if isinstance(raw_scores, dict):
                if "pass_at_1" in raw_scores:
                    pass_values.append(float(raw_scores["pass_at_1"]))
                if "execution_latency_ms" in raw_scores:
                    sandbox_latencies.append(float(raw_scores["execution_latency_ms"]))
                if "syntax_check" in raw_scores:
                    syntax_values.append(float(raw_scores["syntax_check"]))

    if not pass_values:
        results.append(
            _stage(
                ValidationStage.PASS_AT_1_EVALUATION,
                "FAIL",
                latency_ms=(time.perf_counter() - start) * 1000,
                message="pass_at_1 not recorded in scores",
                root_cause="pass@1 metric missing from results",
                recommended_fix="Ensure executable_code_generation task metrics include pass_at_1",
                severity="critical",
            )
        )
    else:
        results.append(
            _stage(
                ValidationStage.PASS_AT_1_EVALUATION,
                "PASS",
                latency_ms=(time.perf_counter() - start) * 1000,
                message=f"pass@1 recorded for {len(pass_values)} cells (mean={sum(pass_values)/len(pass_values):.3f})",
                details={"pass_at_1_values": pass_values},
            )
        )

    start = time.perf_counter()
    if not syntax_values:
        results.append(
            _stage(
                ValidationStage.SYNTAX_EVALUATION,
                "FAIL",
                latency_ms=(time.perf_counter() - start) * 1000,
                message="syntax_check not recorded",
                root_cause="Syntax metric missing",
                recommended_fix="Include syntax_check in task metrics",
                severity="medium",
            )
        )
    else:
        results.append(
            _stage(
                ValidationStage.SYNTAX_EVALUATION,
                "PASS",
                latency_ms=(time.perf_counter() - start) * 1000,
                message=f"syntax_check recorded (mean={sum(syntax_values)/len(syntax_values):.3f})",
            )
        )

    start = time.perf_counter()
    if sandbox_latencies or pass_values:
        results.append(
            _stage(
                ValidationStage.SANDBOX_EXECUTION,
                "PASS",
                latency_ms=(time.perf_counter() - start) * 1000,
                message="Sandbox executed during scoring",
                details={"sandbox_latency_ms_total": sum(sandbox_latencies)},
            )
        )
    else:
        results.append(
            _stage(
                ValidationStage.SANDBOX_EXECUTION,
                "FAIL",
                latency_ms=(time.perf_counter() - start) * 1000,
                message="No evidence of sandbox execution",
                root_cause="Tests not executable or sandbox disabled",
                recommended_fix="Verify executable_code_generation task and sandbox module",
                severity="critical",
            )
        )

    return results


def run_sanity_checks(
    output_dir: Path,
    *,
    temperature: float,
    num_tasks: int,
) -> tuple[dict[str, bool], StageResult]:
    start = time.perf_counter()
    sanity: dict[str, bool] = {
        "predictions_non_empty": False,
        "tests_execute": False,
        "sandbox_isolated": False,
        "prompts_controlled_format": len(verify_prompt_family()) == 0,
        "pass_at_1_not_degenerate": False,
        "temperature_respected": False,
    }

    parquet_path = output_dir / "results.parquet"
    if parquet_path.exists():
        df = read_results(parquet_path)
        completed = df[df["status"] == "completed"]
        if not completed.empty:
            sanity["predictions_non_empty"] = not completed["prediction"].astype(str).str.strip().eq("").any()
            sanity["temperature_respected"] = completed["temperature"].astype(float).eq(temperature).all()
            pass_values: list[float] = []
            for raw_scores in completed.get("scores", pd.Series(dtype=object)):
                if isinstance(raw_scores, dict) and "pass_at_1" in raw_scores:
                    pass_values.append(float(raw_scores["pass_at_1"]))
            if pass_values:
                sanity["tests_execute"] = True
                if num_tasks >= 3 and len(set(pass_values)) > 1:
                    sanity["pass_at_1_not_degenerate"] = True
                elif num_tasks < 3:
                    sanity["pass_at_1_not_degenerate"] = 0.0 < sum(pass_values) / len(pass_values) < 1.0 or len(pass_values) == 1
                else:
                    sanity["pass_at_1_not_degenerate"] = not (all(v == 0.0 for v in pass_values) or all(v == 1.0 for v in pass_values))

    prog = "while True:\n    pass\n"
    res = execute_python_program(prog, limits=ExecutionLimits(timeout_seconds=0.15, memory_mb=64))
    sanity["sandbox_isolated"] = res.timed_out or res.exit_code is not None

    failed = [k for k, v in sanity.items() if not v]
    if failed and num_tasks >= 3 and "pass_at_1_not_degenerate" in failed and len(failed) == 1:
        status: StageStatus = "WARN"
        message = f"Sanity warnings: {failed} (may occur with small samples)"
        severity: Severity = "medium"
    elif failed:
        status = "FAIL"
        message = f"Sanity checks failed: {failed}"
        severity = "high"
    else:
        status = "PASS"
        message = "All sanity checks passed"
        severity = "low"

    return sanity, _stage(
        ValidationStage.SANITY_CHECKS,
        status,
        latency_ms=(time.perf_counter() - start) * 1000,
        message=message,
        severity=severity,
        details=sanity,
    )


def compute_timing(output_dir: Path, pipeline_ms: float) -> TimingBreakdown:
    timing = TimingBreakdown(pipeline_latency_ms=pipeline_ms)
    parquet_path = output_dir / "results.parquet"
    if not parquet_path.exists():
        return timing

    df = read_results(parquet_path)
    completed = df[df["status"] == "completed"]
    timing.observations = len(completed)
    if completed.empty:
        return timing

    timing.model_latency_ms = float(completed["latency_ms"].sum())
    sandbox_total = 0.0
    for raw_scores in completed.get("scores", pd.Series(dtype=object)):
        if isinstance(raw_scores, dict) and "execution_latency_ms" in raw_scores:
            sandbox_total += float(raw_scores["execution_latency_ms"])
    timing.sandbox_latency_ms = sandbox_total
    timing.evaluation_latency_ms = sandbox_total
    timing.io_latency_ms = max(0.0, pipeline_ms - timing.model_latency_ms - timing.evaluation_latency_ms)
    return timing


def run_confirmatory_validation(
    *,
    benchmark: str = "humaneval",
    model: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    temperature: float = 0.0,
    runs: int = 1,
    tasks: int = 3,
    verbose: bool = False,
    skip_experiment: bool = False,
    reference_config: Path | str | None = None,
    expected_total_tasks: int | None = None,
) -> ValidationReport:
    """Execute full pre-flight validation and return structured report."""
    pipeline_start = time.perf_counter()
    stages: list[StageResult] = []
    warnings: list[str] = []
    resolved = resolve_benchmark(benchmark)

    ref_path = Path(reference_config) if reference_config else reference_config_path(benchmark)
    ref_errors = validate_config(ref_path)
    if ref_errors:
        stages.append(
            _stage(
                ValidationStage.BENCHMARK_LOAD,
                "FAIL",
                message=f"Reference config invalid: {ref_errors[0]}",
                root_cause="Invalid configuration",
                recommended_fix=f"Fix {ref_path}",
                severity="critical",
            )
        )
        env = collect_environment()
        env["gpu_summary"] = _gpu_summary(env)
        output_stub = Path(f"experiments/preflight_validate_{resolved}") / "aborted"
        report = ValidationReport(
            benchmark=resolved,
            output_dir=str(output_stub),
            stages=stages,
            timing=TimingBreakdown(),
            environment=env,
            benchmark_info={"name": resolved},
            sanity={},
            warnings=warnings,
            ready_to_launch=False,
        )
        output_stub.mkdir(parents=True, exist_ok=True)
        write_reports(report, output_stub)
        return report

    bench_stage, bench_info = validate_benchmark_load(
        benchmark,
        num_tasks=tasks,
        expected_total_tasks=expected_total_tasks,
    )
    stages.append(bench_stage)
    if bench_stage.failed:
        env = collect_environment()
        env["gpu_summary"] = _gpu_summary(env)
        report = ValidationReport(
            benchmark=resolved,
            output_dir="",
            stages=stages,
            timing=TimingBreakdown(),
            environment=env,
            benchmark_info=bench_info,
            sanity={},
            warnings=warnings,
            ready_to_launch=False,
        )
        return report

    config, reference_path, task_ids = build_preflight_config(
        benchmark,
        model_id=model,
        prompt_id=prompt,
        temperature=temperature,
        number_of_runs=runs,
        num_tasks=tasks,
        reference_config=ref_path,
    )

    bench_info.update(
        {
            "prompt_protocol": PROMPT_PROTOCOL_VERSION,
            "prompt_id": prompt,
            "model_id": model,
            "model_ollama_id": config.models[0].model_id,
            "temperature": temperature,
            "runs": runs,
        }
    )

    path = dataset_path(benchmark)
    dataset = TaskDataset.from_jsonl(path)
    examples_by_id = {record.task_id: record for record in dataset.records}
    if task_ids[0] not in examples_by_id:
        stages.append(
            _stage(
                ValidationStage.TASK_METADATA,
                "FAIL",
                message=f"Selected task '{task_ids[0]}' not in dataset",
                root_cause="Task selection mismatch",
                recommended_fix="Re-materialize benchmark or adjust --tasks",
                severity="critical",
            )
        )
        env = collect_environment()
        env["gpu_summary"] = _gpu_summary(env)
        abort_dir = config.output.directory / "aborted"
        abort_dir.mkdir(parents=True, exist_ok=True)
        report = ValidationReport(
            benchmark=resolved,
            output_dir=str(abort_dir),
            stages=stages,
            timing=TimingBreakdown(),
            environment=env,
            benchmark_info=bench_info,
            sanity={},
            warnings=warnings,
            ready_to_launch=False,
        )
        write_reports(report, abort_dir)
        return report
    example = examples_by_id[task_ids[0]]

    preflight_dir = config.output.directory
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
    output_dir = preflight_dir / timestamp
    config = config.model_copy(update={"output": config.output.model_copy(update={"directory": output_dir})})

    stages.append(validate_task_metadata(benchmark, task_ids))
    stages.append(validate_prompt_generation(prompt, example.input))
    stages.append(validate_model_provider(config, model))
    stages.append(validate_ollama_connectivity(config, model))
    stages.append(validate_timeout_handling())

    free_gb = _disk_free_gb(output_dir.parent)
    if free_gb < 5.0:
        warnings.append(f"Low disk space: {free_gb:.1f} GB free")

    env = collect_environment()
    env["gpu_summary"] = _gpu_summary(env)
    env["disk_free_gb"] = round(free_gb, 2)

    critical_pre = [s for s in stages if s.failed and s.severity in {"critical", "high", None}]
    if critical_pre and not skip_experiment:
        report = ValidationReport(
            benchmark=resolved,
            output_dir=str(output_dir),
            stages=stages,
            timing=TimingBreakdown(),
            environment=env,
            benchmark_info=bench_info,
            sanity={},
            warnings=warnings,
            ready_to_launch=False,
        )
        write_reports(report, output_dir)
        return report

    experiment_ms = 0.0
    if not skip_experiment:
        exp_start = time.perf_counter()
        config_yaml = output_dir / "preflight_config.yaml"
        output_dir.mkdir(parents=True, exist_ok=True)
        import yaml

        config_yaml.write_text(
            yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        runner = ExperimentRunner(
            config,
            config_path=config_yaml,
            dry_run=False,
            auto_resume=False,
        )
        manifest = runner.run()
        output_dir = runner.output_dir
        experiment_ms = (time.perf_counter() - exp_start) * 1000
        if manifest.status != "completed":
            warnings.append(f"Experiment finished with status={manifest.status}")
        if manifest.failed_cells:
            warnings.append(f"{manifest.failed_cells} cells failed during pre-flight run")

        stages.extend(_validate_inference_and_metrics(output_dir))
        stages.extend(_validate_post_run_artifacts(output_dir, config))
        stages.append(validate_resume_mechanism(output_dir))
    else:
        stages.append(
            _stage(
                ValidationStage.INFERENCE,
                "SKIP",
                message="Experiment skipped (dry validation mode)",
                severity="low",
            )
        )

    pipeline_ms = (time.perf_counter() - pipeline_start) * 1000
    timing = compute_timing(output_dir, pipeline_ms if not skip_experiment else experiment_ms)
    if skip_experiment:
        timing = TimingBreakdown()

    sanity, sanity_stage = run_sanity_checks(output_dir, temperature=temperature, num_tasks=tasks)
    stages.append(sanity_stage)

    critical_failures = [s for s in stages if s.failed and s.severity in {"critical", "high", None}]
    ready = not critical_failures and bench_stage.passed

    report = ValidationReport(
        benchmark=resolved,
        output_dir=str(output_dir.resolve()),
        stages=stages,
        timing=timing,
        environment=env,
        benchmark_info=bench_info,
        sanity=sanity,
        warnings=warnings,
        ready_to_launch=ready,
    )
    write_reports(report, output_dir)

    if verbose:
        for stage in stages:
            print(f"[{stage.status}] {stage.stage.value}: {stage.message}")

    return report
