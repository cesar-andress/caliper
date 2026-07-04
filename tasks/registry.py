"""Task type registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from caliper.tasks.base import BaseTask
from caliper.tasks.loader import TaskDataset
from caliper.tasks.schema import TaskDomain

_TASKS: dict[TaskDomain, type[BaseTask]] = {}


def register_task(domain: TaskDomain):
    """Decorator to register a task class for a domain."""

    def decorator(cls: type[BaseTask]) -> type[BaseTask]:
        cls.domain = domain
        _TASKS[domain] = cls
        return cls

    return decorator


def get_task_class(domain: TaskDomain) -> type[BaseTask]:
    if domain not in _TASKS:
        registered = ", ".join(sorted(_TASKS)) or "(none)"
        msg = f"Unknown task domain '{domain}'. Registered: {registered}"
        raise KeyError(msg)
    return _TASKS[domain]


def list_task_domains() -> list[TaskDomain]:
    return sorted(_TASKS)


def create_task(
    domain: TaskDomain,
    task_id: str,
    dataset: TaskDataset | Path | str,
    **config: Any,
) -> BaseTask:
    """Instantiate a registered task by domain."""
    cls = get_task_class(domain)
    return cls(task_id, dataset, config=config)
