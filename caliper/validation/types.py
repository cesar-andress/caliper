"""Pre-flight validation types for confirmatory experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

StageStatus = Literal["PASS", "FAIL", "WARN", "SKIP"]
Severity = Literal["critical", "high", "medium", "low"]


class ValidationStage(str, Enum):
    BENCHMARK_LOAD = "benchmark_load"
    TASK_METADATA = "task_metadata"
    PROMPT_GENERATION = "prompt_generation"
    MODEL_PROVIDER = "model_provider"
    OLLAMA_CONNECTIVITY = "ollama_connectivity"
    INFERENCE = "inference"
    CODE_EXTRACTION = "code_extraction"
    SANDBOX_EXECUTION = "sandbox_execution"
    TIMEOUT_HANDLING = "timeout_handling"
    PASS_AT_1_EVALUATION = "pass_at_1_evaluation"
    SYNTAX_EVALUATION = "syntax_evaluation"
    STATISTICAL_DATASET = "statistical_dataset"
    PARQUET_EXPORT = "parquet_export"
    MANIFEST_GENERATION = "manifest_generation"
    ARTIFACT_EXPORT = "artifact_export"
    REPORT_GENERATION = "report_generation"
    ROBUSTNESS_PIPELINE = "robustness_pipeline"
    SANITY_CHECKS = "sanity_checks"
    RESUME_MECHANISM = "resume_mechanism"


@dataclass
class StageResult:
    """Outcome of one validation stage."""

    stage: ValidationStage
    status: StageStatus
    latency_ms: float = 0.0
    message: str = ""
    root_cause: str | None = None
    recommended_fix: str | None = None
    severity: Severity | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    @property
    def failed(self) -> bool:
        return self.status == "FAIL"


@dataclass
class TimingBreakdown:
    """Pipeline latency measurements in milliseconds."""

    model_latency_ms: float = 0.0
    evaluation_latency_ms: float = 0.0
    sandbox_latency_ms: float = 0.0
    io_latency_ms: float = 0.0
    pipeline_latency_ms: float = 0.0
    observations: int = 0

    def per_observation_ms(self) -> float:
        if self.observations <= 0:
            return 0.0
        return self.pipeline_latency_ms / self.observations

    def estimate_hours(self, n_observations: int) -> float:
        return (self.per_observation_ms() * n_observations) / 3_600_000.0


@dataclass
class ValidationReport:
    """Full pre-flight validation outcome."""

    benchmark: str
    output_dir: str
    stages: list[StageResult]
    timing: TimingBreakdown
    environment: dict[str, Any]
    benchmark_info: dict[str, Any]
    sanity: dict[str, Any]
    warnings: list[str]
    ready_to_launch: bool

    def status_table(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for stage in self.stages:
            rows.append(
                {
                    "stage": stage.stage.value,
                    "status": stage.status,
                    "latency_ms": f"{stage.latency_ms:.1f}",
                    "message": stage.message,
                }
            )
        return rows

    @property
    def all_passed(self) -> bool:
        return all(s.status in {"PASS", "WARN", "SKIP"} for s in self.stages) and not any(
            s.status == "FAIL" and s.severity in {"critical", "high", None} for s in self.stages
        )

    @property
    def failures(self) -> list[StageResult]:
        return [s for s in self.stages if s.failed]
