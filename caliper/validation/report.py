"""Markdown report generation for pre-flight validation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from caliper.validation.types import TimingBreakdown, ValidationReport, ValidationStage


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_validation_report(report: ValidationReport) -> str:
    """Render validation_report.md content."""
    now = datetime.now(tz=UTC).isoformat()
    env = report.environment
    bench = report.benchmark_info
    timing = report.timing

    stage_rows = [
        [s.stage.value, s.status, f"{s.latency_ms:.1f}", s.message]
        for s in report.stages
        if s.stage != ValidationStage.SANITY_CHECKS
    ]

    runtime_rows = [
        ["1,000", f"{timing.estimate_hours(1000):.2f}"],
        ["5,000", f"{timing.estimate_hours(5000):.2f}"],
        ["10,000", f"{timing.estimate_hours(10000):.2f}"],
        ["20,000", f"{timing.estimate_hours(20000):.2f}"],
    ]

    failure_section = ""
    failures = report.failures
    if failures:
        failure_lines = []
        for item in failures:
            failure_lines.append(f"### {item.stage.value}")
            failure_lines.append(f"- **Root cause:** {item.root_cause or item.message}")
            failure_lines.append(f"- **Recommended fix:** {item.recommended_fix or 'See logs.'}")
            failure_lines.append(f"- **Severity:** {item.severity or 'critical'}")
            failure_lines.append("")
        failure_section = "## Failure diagnostics\n\n" + "\n".join(failure_lines)

    sanity = report.sanity
    sanity_rows = [[k, "PASS" if v else "FAIL"] for k, v in sorted(sanity.items())]

    warnings_section = ""
    if report.warnings:
        warnings_section = "## Warnings\n\n" + "\n".join(f"- {w}" for w in report.warnings) + "\n"

    launch_status = "**READY TO LAUNCH**" if report.ready_to_launch else "**NOT READY — resolve failures first**"

    return f"""# CALIPER Confirmatory Pre-flight Validation Report

Generated: {now}

Overall status: {launch_status}

Benchmark: `{report.benchmark}`

Output directory: `{report.output_dir}`

## Environment

| Field | Value |
| --- | --- |
| Python | {env.get('python_version', 'unknown')} |
| CALIPER | {env.get('software_version', 'unknown')} |
| Git commit | {env.get('git_commit', 'unknown')} |
| OS | {env.get('os', {}).get('system', 'unknown')} {env.get('os', {}).get('release', 'unknown')} |
| CPU | {env.get('cpu', {}).get('processor', 'unknown')} |
| GPU | {env.get('gpu_summary', 'unknown')} |

## Benchmark provenance

| Field | Value |
| --- | --- |
| Benchmark | {bench.get('name', report.benchmark)} |
| Version | {bench.get('version', 'unknown')} |
| Dataset path | `{bench.get('dataset_path', '')}` |
| SHA-256 | `{bench.get('checksum', 'unknown')}` |
| Tasks in pool | {bench.get('num_tasks', 'unknown')} |
| Tasks exercised | {bench.get('tasks_exercised', 'unknown')} |

## Prompt protocol

| Field | Value |
| --- | --- |
| Protocol version | {bench.get('prompt_protocol', 'unknown')} |
| Prompt variant | {bench.get('prompt_id', 'unknown')} |
| Model | {bench.get('model_id', 'unknown')} (`{bench.get('model_ollama_id', '')}`) |
| Temperature | {bench.get('temperature', 'unknown')} |
| Runs | {bench.get('runs', 'unknown')} |

## Pipeline timings

| Component | Latency (ms) |
| --- | --- |
| Model inference (total) | {timing.model_latency_ms:.1f} |
| Evaluation (total) | {timing.evaluation_latency_ms:.1f} |
| Sandbox execution (total) | {timing.sandbox_latency_ms:.1f} |
| I/O & finalization (total) | {timing.io_latency_ms:.1f} |
| End-to-end pipeline | {timing.pipeline_latency_ms:.1f} |
| Observations | {timing.observations} |
| Per-observation mean | {timing.per_observation_ms():.1f} ms |

## Estimated runtime (confirmatory scale)

{_format_table(["Observations", "Estimated hours"], runtime_rows)}

Full confirmatory study (9,600 cells per benchmark): **{timing.estimate_hours(9600):.1f} h** per benchmark at current throughput.

## Stage checklist

{_format_table(["Stage", "Status", "Latency (ms)", "Message"], stage_rows)}

## Sanity checks

{_format_table(["Check", "Status"], sanity_rows)}

{warnings_section}
{failure_section}
"""


def render_launch_checklist(report: ValidationReport, *, full_study_cells: int = 9600) -> str:
    """Render launch_checklist.md for the 24-hour confirmatory run."""
    env = report.environment
    bench = report.benchmark_info
    timing = report.timing
    gpu = env.get("gpu", {})
    gpu_ok = bool(gpu.get("gpu_available") or gpu.get("device_name"))

    def checked(stage: ValidationStage) -> str:
        result = next((s for s in report.stages if s.stage == stage), None)
        if result is None:
            return "[ ]"
        return "[x]" if result.status == "PASS" else "[ ]"

    def sanity_checked(key: str) -> str:
        return "[x]" if report.sanity.get(key) else "[ ]"

    ollama_ok = checked(ValidationStage.OLLAMA_CONNECTIVITY) == "[x]"
    model_ok = checked(ValidationStage.MODEL_PROVIDER) == "[x]"
    bench_ok = checked(ValidationStage.BENCHMARK_LOAD) == "[x]"
    artifact_ok = checked(ValidationStage.ARTIFACT_EXPORT) == "[x]"
    robust_ok = checked(ValidationStage.ROBUSTNESS_PIPELINE) == "[x]"
    resume_ok = checked(ValidationStage.RESUME_MECHANISM) == "[x]"

    return f"""# Confirmatory Study Launch Checklist

Generated from pre-flight validation on `{report.benchmark}`.

Estimated runtime at measured throughput: **{timing.estimate_hours(full_study_cells):.1f} hours** for {full_study_cells:,} observations.

## Infrastructure

- {'[x]' if ollama_ok else '[ ]'} Ollama running and reachable
- {'[x]' if model_ok else '[ ]'} Required model downloaded (`{bench.get('model_ollama_id', '')}`)
- {'[x]' if gpu_ok else '[ ]'} GPU detected ({env.get('gpu_summary', 'none')})
- [ ] Enough disk space (recommend ≥ 50 GB free for artifacts + checkpoints)
- [ ] Output directory empty or resume plan documented (`experiments/paper1_confirmatory_{report.benchmark.replace('_plus', '')}`)

## Data integrity

- {'[x]' if bench_ok else '[ ]'} HumanEval+ checksum verified (`{bench.get('checksum', '')}`) — when running humaneval study
- {'[x]' if bench_ok else '[ ]'} MBPP checksum verified — when running mbpp study
- [ ] Confirmatory YAML validated: `caliper validate -c configs/paper1/confirmatory_humaneval.yaml`
- [ ] Confirmatory YAML validated: `caliper validate -c configs/paper1/confirmatory_mbpp.yaml`

## Pipeline verification (pre-flight)

- {checked(ValidationStage.BENCHMARK_LOAD)} Benchmark loads correctly
- {checked(ValidationStage.INFERENCE)} Live inference succeeded (not mocked)
- {checked(ValidationStage.SANDBOX_EXECUTION)} Sandbox execution succeeded
- {checked(ValidationStage.PASS_AT_1_EVALUATION)} pass@1 evaluation succeeded
- {checked(ValidationStage.STATISTICAL_DATASET)} Statistical dataset generated
- {checked(ValidationStage.PARQUET_EXPORT)} Parquet export succeeded
- {checked(ValidationStage.MANIFEST_GENERATION)} Manifest generated
- {artifact_ok} Artifact export tested
- {checked(ValidationStage.REPORT_GENERATION)} Report generation tested
- {robust_ok} Robustness pipeline tested
- {resume_ok} Resume mechanism tested

## Sanity checks

- {sanity_checked('predictions_non_empty')} Predictions are non-empty
- {sanity_checked('tests_execute')} Unit tests actually execute in sandbox
- {sanity_checked('sandbox_isolated')} Sandbox isolates execution
- {sanity_checked('prompts_controlled_format')} All prompts preserve required output format
- {sanity_checked('pass_at_1_not_degenerate')} pass@1 is neither always 0 nor always 1 (when N≥3)
- {sanity_checked('temperature_respected')} Temperature recorded correctly in results

## Launch decision

- {'[x]' if report.ready_to_launch else '[ ]'} Pre-flight validation PASS — safe to launch 24-hour confirmatory run
- [ ] Estimated runtime confirmed ({timing.estimate_hours(full_study_cells):.1f} h per 9,600-cell study)
- [ ] Pilot results archived as supplementary material (unchanged)

## Launch commands (manual — do not auto-run)

```bash
# HumanEval+ confirmatory study
caliper run -c configs/paper1/confirmatory_humaneval.yaml

# MBPP confirmatory study
caliper run -c configs/paper1/confirmatory_mbpp.yaml

# Post-run analysis
make paper1-confirmatory-analysis
make paper1-confirmatory-robustness
```
"""


def write_reports(report: ValidationReport, output_dir: Path) -> dict[str, Path]:
    """Write validation_report.md and launch_checklist.md."""
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = output_dir / "validation_report.md"
    checklist_path = output_dir / "launch_checklist.md"
    validation_path.write_text(render_validation_report(report), encoding="utf-8")
    checklist_path.write_text(render_launch_checklist(report), encoding="utf-8")
    return {"validation_report": validation_path, "launch_checklist": checklist_path}
