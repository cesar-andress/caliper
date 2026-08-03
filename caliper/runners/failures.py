"""Terminal failure tracking for factorial experiment cells."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from caliper.runners.results import ExperimentResultRecord


class FailureWriter:
    """Append-only log of terminal cell failures."""

    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / "failures.jsonl"

    def append(self, record: ExperimentResultRecord) -> None:
        """Persist one terminal failure event."""
        if record.status != "failed":
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = record.model_dump()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str))
            handle.write("\n")


def load_results_records(jsonl_path: Path) -> list[ExperimentResultRecord]:
    """Load every result record from JSONL."""
    if not jsonl_path.exists():
        return []

    records: list[ExperimentResultRecord] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(ExperimentResultRecord.model_validate(json.loads(line)))
    return records


def latest_records_by_cell(records: list[ExperimentResultRecord]) -> dict[str, ExperimentResultRecord]:
    """Return the latest record per cell_id ordered by executed_at."""
    grouped: dict[str, list[ExperimentResultRecord]] = defaultdict(list)
    for record in records:
        grouped[record.cell_id].append(record)

    latest: dict[str, ExperimentResultRecord] = {}
    for cell_id, cell_records in grouped.items():
        latest[cell_id] = sorted(cell_records, key=lambda item: item.executed_at)[-1]
    return latest


def count_terminal_failures(jsonl_path: Path) -> int:
    """Count cells whose latest result record is terminal failed."""
    latest = latest_records_by_cell(load_results_records(jsonl_path))
    return sum(1 for record in latest.values() if record.status == "failed")


def count_terminal_completions(jsonl_path: Path) -> int:
    """Count cells whose latest result record completed successfully."""
    latest = latest_records_by_cell(load_results_records(jsonl_path))
    return sum(1 for record in latest.values() if record.status == "completed")


def duplicate_cell_ids(records: list[ExperimentResultRecord]) -> list[str]:
    """Return cell IDs that appear more than once in the raw result log."""
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[record.cell_id] += 1
    return sorted(cell_id for cell_id, count in counts.items() if count > 1)
