"""Abstract interface for experiment result storage."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class ResultStore(ABC):
    """Persist and retrieve experiment results in structured formats."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    @abstractmethod
    def save(self, df: pd.DataFrame, name: str) -> Path:
        """Save a DataFrame and return the written path."""

    @abstractmethod
    def load(self, name: str) -> pd.DataFrame:
        """Load a previously saved result set."""

    @abstractmethod
    def save_manifest(self, manifest: dict[str, Any]) -> Path:
        """Save run metadata/manifest."""

    @abstractmethod
    def list_runs(self) -> list[str]:
        """List available run IDs in the store."""
