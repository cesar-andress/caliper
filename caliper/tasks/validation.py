"""Task record and dataset validation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from caliper.tasks.schema import TaskDomain, TaskMetadata

if TYPE_CHECKING:
    from caliper.tasks.loader import TaskDataset


class TaskValidationError(Exception):
    """Raised when task metadata or a dataset fails validation."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(format_task_errors(errors))


def format_task_errors(errors: list[str]) -> str:
    body = "\n".join(f"  - {err}" for err in errors)
    return f"Task validation failed:\n{body}"


def validate_task_record(data: dict, *, expected_domain: TaskDomain | None = None) -> TaskMetadata:
    """Validate a single task record dict and return TaskMetadata.

    Raises:
        TaskValidationError: If validation fails.
    """
    errors = collect_task_record_errors(data, expected_domain=expected_domain)
    if errors:
        raise TaskValidationError(errors)
    return TaskMetadata.model_validate(data)


def collect_task_record_errors(
    data: dict,
    *,
    expected_domain: TaskDomain | None = None,
    line_number: int | None = None,
) -> list[str]:
    """Return validation errors for a single record without raising."""
    prefix = f"line {line_number}: " if line_number is not None else ""
    errors: list[str] = []

    try:
        record = TaskMetadata.model_validate(data)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(part) for part in err["loc"])
            errors.append(f"{prefix}{loc}: {err['msg']}")
        return errors

    if expected_domain is not None and record.domain != expected_domain:
        errors.append(
            f"{prefix}domain: expected '{expected_domain}', got '{record.domain}'"
        )

    return errors


def validate_dataset(
    dataset: TaskDataset,
    *,
    expected_domain: TaskDomain | None = None,
) -> list[str]:
    """Validate all records in a dataset including cross-record checks."""
    errors: list[str] = []
    seen_ids: set[str] = set()

    for line_number, record in enumerate(dataset.records, start=1):
        if expected_domain is not None and record.domain != expected_domain:
            errors.append(
                f"line {line_number}: domain: expected '{expected_domain}', "
                f"got '{record.domain}'"
            )
        if record.task_id in seen_ids:
            errors.append(f"line {line_number}: duplicate task_id '{record.task_id}'")
        seen_ids.add(record.task_id)

    return errors


def validate_dataset_file(
    path: Path,
    *,
    expected_domain: TaskDomain | None = None,
) -> list[str]:
    """Load and validate a JSONL dataset file, returning all errors."""
    from caliper.tasks.loader import TaskDataset

    errors: list[str] = []
    if not path.exists():
        return [f"dataset file not found: {path}"]

    try:
        dataset = TaskDataset.from_jsonl(path, validate_lines=False)
    except TaskValidationError as exc:
        return list(exc.errors)

    errors.extend(validate_dataset(dataset, expected_domain=expected_domain))
    return errors
