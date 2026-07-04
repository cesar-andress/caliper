"""Execution progress tracking with ETA and throughput."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ExecutionProgress:
    """Track factorial experiment execution progress."""

    total_cells: int
    completed_cells: int = 0
    failed_cells: int = 0
    skipped_cells: int = 0
    _started_at: float = field(default_factory=time.monotonic)
    _processed_cells: int = 0

    @property
    def pending_cells(self) -> int:
        return max(
            self.total_cells - self.completed_cells - self.failed_cells - self.skipped_cells,
            0,
        )

    @property
    def elapsed_seconds(self) -> float:
        return max(time.monotonic() - self._started_at, 0.0)

    @property
    def throughput_cells_per_second(self) -> float:
        if self._processed_cells == 0 or self.elapsed_seconds <= 0:
            return 0.0
        return self._processed_cells / self.elapsed_seconds

    @property
    def eta_seconds(self) -> float | None:
        remaining = self.total_cells - self.skipped_cells - self._processed_cells
        if remaining <= 0:
            return 0.0
        rate = self.throughput_cells_per_second
        if rate <= 0:
            return None
        return remaining / rate

    def record_skip(self) -> None:
        self.skipped_cells += 1

    def record_completion(self, *, success: bool) -> None:
        self._processed_cells += 1
        if success:
            self.completed_cells += 1
        else:
            self.failed_cells += 1

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "total_cells": self.total_cells,
            "completed_cells": self.completed_cells,
            "failed_cells": self.failed_cells,
            "skipped_cells": self.skipped_cells,
            "pending_cells": self.pending_cells,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "throughput_cells_per_second": round(self.throughput_cells_per_second, 4),
            "eta_seconds": round(self.eta_seconds, 3) if self.eta_seconds is not None else None,
        }
