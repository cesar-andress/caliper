"""Task metadata schema for benchmark evaluation."""

from __future__ import annotations

import re
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, field_validator

TaskDomain = Literal["code_generation", "bug_repair", "code_summarization"]
Difficulty = Literal["easy", "medium", "hard"]

TASK_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


class TaskMetadata(BaseModel):
    """Schema for a single benchmark task instance."""

    task_id: str
    domain: TaskDomain
    input: str
    expected_output: str | None = None
    tests: list[str] | None = None
    language: str
    source_benchmark: str
    tags: list[str] = Field(default_factory=list)
    difficulty: Difficulty | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, v: str) -> str:
        if not TASK_ID_PATTERN.match(v):
            msg = (
                "task_id must start with a lowercase letter and contain only "
                "lowercase letters, digits, underscores, or hyphens"
            )
            raise ValueError(msg)
        return v

    @field_validator("input")
    @classmethod
    def _validate_input_not_empty(cls, v: str) -> str:
        if not v.strip():
            msg = "input must not be empty"
            raise ValueError(msg)
        return v

    @field_validator("language")
    @classmethod
    def _validate_language(cls, v: str) -> str:
        if not v.strip():
            msg = "language must not be empty"
            raise ValueError(msg)
        return v.strip().lower()

    @field_validator("tags")
    @classmethod
    def _normalize_tags(cls, v: list[str]) -> list[str]:
        return [tag.strip() for tag in v if tag.strip()]

    @field_validator("tests")
    @classmethod
    def _validate_tests(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        cleaned = [t.strip() for t in v if t.strip()]
        return cleaned or None

    def to_example_dict(self) -> dict[str, Any]:
        """Serialize to a flat dict for logging and storage."""
        return self.model_dump()
