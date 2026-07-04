"""Synthetic result matrices for ranking fragility tests."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def generate_stable_ranking_data(
    *,
    n_models: int = 4,
    n_tasks: int = 20,
    n_prompts: int = 2,
    n_runs: int = 3,
    separation: float = 0.15,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate results where model rankings are clearly separated and stable."""
    rng = np.random.default_rng(seed)
    models = [f"model_{i}" for i in range(n_models)]
    base_scores = {m: 0.95 - i * separation for i, m in enumerate(models)}

    rows: list[dict[str, object]] = []
    for model, task, prompt, run in itertools.product(
        models,
        [f"task_{i}" for i in range(n_tasks)],
        [f"prompt_{i}" for i in range(n_prompts)],
        range(n_runs),
    ):
        noise = rng.normal(0, 0.005)
        rows.append(
            {
                "model": model,
                "task_id": task,
                "prompt_id": prompt,
                "run_id": run,
                "metric_name": "exact_match",
                "metric_value": float(np.clip(base_scores[model] + noise, 0, 1)),
            }
        )
    return pd.DataFrame(rows)


def generate_unstable_ranking_data(
    *,
    n_models: int = 4,
    n_tasks: int = 20,
    n_prompts: int = 2,
    n_runs: int = 3,
    seed: int = 1,
) -> pd.DataFrame:
    """Generate results where models have overlapping task-level scores."""
    rng = np.random.default_rng(seed)
    models = [f"model_{i}" for i in range(n_models)]
    tasks = [f"task_{i}" for i in range(n_tasks)]

    # Each model wins on different task subsets → fragile aggregate ranking.
    task_winners = {t: models[i % n_models] for i, t in enumerate(tasks)}

    rows: list[dict[str, object]] = []
    for model, task, prompt, run in itertools.product(
        models,
        tasks,
        [f"prompt_{i}" for i in range(n_prompts)],
        range(n_runs),
    ):
        if task_winners[task] == model:
            value = 0.85 + rng.normal(0, 0.02)
        else:
            value = 0.80 + rng.normal(0, 0.04)
        rows.append(
            {
                "model": model,
                "task_id": task,
                "prompt_id": prompt,
                "run_id": run,
                "metric_name": "exact_match",
                "metric_value": float(np.clip(value, 0, 1)),
            }
        )
    return pd.DataFrame(rows)
