"""Persistent per-cell checkpoints for restartable experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from caliper.runners.results import ExperimentResultRecord


class CheckpointStore:
    """Write and load per-cell checkpoint files."""

    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir

    def ensure_dir(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, cell_id: str) -> Path:
        return self.checkpoint_dir / f"{cell_id}.json"

    def write(self, record: ExperimentResultRecord) -> Path:
        """Persist a cell checkpoint for completed or terminal failed cells."""
        self.ensure_dir()
        path = self.path_for(record.cell_id)
        payload = record.model_dump()
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def write_failed(self, record: ExperimentResultRecord) -> Path:
        """Persist a terminal failed cell checkpoint with error metadata."""
        if record.status != "failed":
            msg = "write_failed requires a failed ExperimentResultRecord"
            raise ValueError(msg)
        return self.write(record)

    def load_completed_cell_ids(self) -> set[str]:
        """Return cell IDs with successful checkpoints."""
        if not self.checkpoint_dir.exists():
            return set()

        completed: set[str] = set()
        for path in self.checkpoint_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("status") in {"completed", "budget_exhausted"}:
                    completed.add(str(data["cell_id"]))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        return completed

    def load_failed_cell_ids(self) -> set[str]:
        """Return cell IDs with terminal failed checkpoints."""
        if not self.checkpoint_dir.exists():
            return set()

        failed: set[str] = set()
        for path in self.checkpoint_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("status") == "failed":
                    failed.add(str(data["cell_id"]))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        return failed

    def load_state(self) -> dict[str, Any]:
        """Load aggregate checkpoint state for resume."""
        state_path = self.checkpoint_dir.parent / "checkpoint_state.json"
        if not state_path.exists():
            return {}
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def save_state(self, state: dict[str, Any]) -> Path:
        """Persist aggregate checkpoint state."""
        state_path = self.checkpoint_dir.parent / "checkpoint_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        return state_path
