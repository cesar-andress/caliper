"""Structured experiment result records and incremental persistence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from caliper.storage.formats import write_manifest, write_results

# budget_exhausted: provider returned empty visible text after spending the
# shared thinking+response token budget (typically done_reason=length).
CellStatus = Literal["completed", "failed", "skipped", "budget_exhausted"]

# Statuses that count as finished for resume / checkpoint skip.
FINISHED_STATUSES = frozenset({"completed", "skipped", "budget_exhausted"})


class ExperimentResultRecord(BaseModel):
    """One row in the experiment results table (one factorial cell)."""

    cell_id: str
    experiment_id: str
    run_id: str
    run_index: int
    model_id: str
    provider_name: str
    provider_type: str
    task_id: str
    prompt_variant_id: str
    temperature: float
    seed: int | None = None
    metric: str
    score: float
    scores: dict[str, float] = Field(default_factory=dict)
    prediction: str = ""
    num_examples: int = 0
    latency_ms: float = 0.0
    status: CellStatus
    error: str | None = None
    done_reason: str | None = None
    eval_count: int | None = None
    prompt_eval_count: int | None = None
    thinking_length: int = 0
    thinking_sha256: str | None = None
    # Full thinking text may be large; prefer length+hash + raw payload on disk.
    thinking: str | None = None
    executed_at: str = Field(
        default_factory=lambda: datetime.now(tz=UTC).isoformat(),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResultWriter:
    """Append-only JSONL writer with resume support and final Parquet export."""

    def __init__(self, output_dir: Path, experiment_id: str, run_id: str) -> None:
        self.output_dir = output_dir
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.jsonl_path = output_dir / "results.jsonl"
        self.parquet_path = output_dir / "results.parquet"
        self._completed_cell_ids: set[str] = set()
        self._records: list[ExperimentResultRecord] = []

    def load_existing(self) -> set[str]:
        """Load finished cell IDs from an existing results file."""
        if not self.jsonl_path.exists():
            return set()

        finished: set[str] = set()
        with self.jsonl_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                record = ExperimentResultRecord.model_validate(data)
                self._records.append(record)
                if record.status in FINISHED_STATUSES:
                    finished.add(record.cell_id)
        self._completed_cell_ids = finished
        return finished

    def load_all_records(self) -> list[ExperimentResultRecord]:
        """Return all records loaded from disk and memory."""
        if not self._records and self.jsonl_path.exists():
            self.load_existing()
        return list(self._records)

    def is_completed(self, cell_id: str) -> bool:
        return cell_id in self._completed_cell_ids

    def append(self, record: ExperimentResultRecord) -> None:
        """Append one result record to JSONL."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json())
            handle.write("\n")
        self._records.append(record)
        if record.status in FINISHED_STATUSES:
            self._completed_cell_ids.add(record.cell_id)

    def finalize(self) -> dict[str, Path]:
        """Write Parquet snapshot and return paths to result files."""
        if not self._records:
            return {}

        df = pd.DataFrame([record.model_dump() for record in self._records])
        write_results(df, self.parquet_path, fmt="parquet")
        return {
            "jsonl": self.jsonl_path,
            "parquet": self.parquet_path,
        }


def write_run_manifest(output_dir: Path, manifest: dict[str, Any]) -> Path:
    return write_manifest(manifest, output_dir / "manifest.json")
