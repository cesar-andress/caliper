"""JSONL dataset loading for benchmark tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from caliper.tasks.schema import TaskMetadata
from caliper.tasks.validation import TaskValidationError, collect_task_record_errors


@dataclass
class TaskDataset:
    """In-memory collection of validated task records loaded from JSONL."""

    records: list[TaskMetadata]
    source_path: Path | None = None

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[TaskMetadata]:
        return iter(self.records)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        validate_lines: bool = True,
        limit: int | None = None,
    ) -> TaskDataset:
        """Load a JSONL file into a TaskDataset.

        Args:
            path: Path to the JSONL file.
            validate_lines: If True, raise on the first invalid line.
            limit: Optional maximum number of records to load.

        Raises:
            FileNotFoundError: If the file does not exist.
            TaskValidationError: If a line fails validation.
        """
        file_path = Path(path)
        if not file_path.exists():
            msg = f"Dataset file not found: {file_path}"
            raise FileNotFoundError(msg)

        records: list[TaskMetadata] = []
        errors: list[str] = []

        with file_path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                if limit is not None and len(records) >= limit:
                    break

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    msg = f"line {line_number}: invalid JSON: {exc.msg}"
                    if validate_lines:
                        raise TaskValidationError([msg]) from exc
                    errors.append(msg)
                    continue

                if not isinstance(data, dict):
                    msg = f"line {line_number}: each record must be a JSON object"
                    if validate_lines:
                        raise TaskValidationError([msg])
                    errors.append(msg)
                    continue

                line_errors = collect_task_record_errors(data, line_number=line_number)
                if line_errors:
                    if validate_lines:
                        raise TaskValidationError(line_errors)
                    errors.extend(line_errors)
                    continue

                records.append(TaskMetadata.model_validate(data))

        if errors and not validate_lines:
            raise TaskValidationError(errors)

        return cls(records=records, source_path=file_path.resolve())

    def ids(self) -> list[str]:
        return [record.task_id for record in self.records]
