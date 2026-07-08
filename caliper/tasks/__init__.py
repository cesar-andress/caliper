"""Benchmark task definitions."""

from caliper.tasks.base import BaseTask, TaskResult
from caliper.tasks.bug_repair import BugRepairTask
from caliper.tasks.code_generation import CodeGenerationTask
from caliper.tasks.code_summarization import CodeSummarizationTask
from caliper.tasks.executable_code_generation import ExecutableCodeGenerationTask
from caliper.tasks.loader import TaskDataset
from caliper.tasks.registry import create_task, get_task_class, list_task_domains, register_task
from caliper.tasks.schema import TaskMetadata
from caliper.tasks.validation import (
    TaskValidationError,
    validate_dataset,
    validate_dataset_file,
    validate_task_record,
)

# Register built-in tasks on import.
from caliper.tasks import bug_repair as _bug_repair  # noqa: F401
from caliper.tasks import code_generation as _code_generation  # noqa: F401
from caliper.tasks import code_summarization as _code_summarization  # noqa: F401
from caliper.tasks import executable_code_generation as _executable_code_generation  # noqa: F401

# Backward-compatible alias.
Task = BaseTask
TaskExample = TaskMetadata

__all__ = [
    "BaseTask",
    "BugRepairTask",
    "CodeGenerationTask",
    "CodeSummarizationTask",
    "ExecutableCodeGenerationTask",
    "Task",
    "TaskDataset",
    "TaskExample",
    "TaskMetadata",
    "TaskResult",
    "TaskValidationError",
    "create_task",
    "get_task_class",
    "list_task_domains",
    "register_task",
    "validate_dataset",
    "validate_dataset_file",
    "validate_task_record",
]
