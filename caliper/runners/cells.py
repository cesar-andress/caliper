"""Factorial cell expansion and identification."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from caliper.config.schema import ExperimentCombination, ExperimentConfig


def expand_cells(config: ExperimentConfig) -> list[ExperimentCombination]:
    """Expand the full factorial design in deterministic order.

    When ``execution.shuffle`` is enabled, cells are shuffled with a RNG
    seeded by ``random_seed`` so expansion is reproducible.
    """
    cells = list(config.iter_combinations())
    if config.execution.shuffle:
        rng = random.Random(config.random_seed)
        rng.shuffle(cells)
    return cells


def make_cell_id(config: ExperimentConfig, cell: ExperimentCombination) -> str:
    """Return a stable identifier for a factorial cell (full deterministic hash)."""
    from caliper.runners.reproducibility import make_cell_hash

    return make_cell_hash(config, cell)


def cell_to_dict(config: ExperimentConfig, cell: ExperimentCombination) -> dict[str, object]:
    """Serialize a cell with its stable identifier."""
    return {
        "cell_id": make_cell_id(config, cell),
        "run_index": cell.run_index,
        "model_id": cell.model_id,
        "provider": cell.provider,
        "task_id": cell.task_id,
        "prompt_variant_id": cell.prompt_variant_id,
        "temperature": cell.temperature,
    }
