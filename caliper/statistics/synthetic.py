"""Synthetic data generation for Paper 1 statistical tests."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def generate_synthetic_results(
    *,
    n_models: int = 2,
    n_tasks: int = 5,
    n_prompts: int = 2,
    n_runs: int = 3,
    model_variance: float = 0.04,
    task_variance: float = 0.09,
    prompt_variance: float = 0.01,
    run_variance: float = 0.0025,
    temperature_variance: float = 0.0,
    residual_variance: float = 0.04,
    grand_mean: float = 0.75,
    seed: int = 42,
    metric_name: str = "exact_match",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Generate a balanced factorial results table with known variance structure.

    Returns:
        Tuple of (DataFrame, true variance components dict).
    """
    rng = np.random.default_rng(seed)
    models = [f"model_{i}" for i in range(n_models)]
    tasks = [f"task_{i}" for i in range(n_tasks)]
    prompts = [f"prompt_{i}" for i in range(n_prompts)]
    runs = list(range(n_runs))
    temperatures = [0.0]

    model_effects = {m: rng.normal(0.0, np.sqrt(model_variance)) for m in models}
    task_effects = {t: rng.normal(0.0, np.sqrt(task_variance)) for t in tasks}
    prompt_effects = {p: rng.normal(0.0, np.sqrt(prompt_variance)) for p in prompts}
    run_effects = {r: rng.normal(0.0, np.sqrt(run_variance)) for r in runs}

    rows: list[dict[str, object]] = []
    for model, task, prompt, run, temp in itertools.product(
        models, tasks, prompts, runs, temperatures
    ):
        value = (
            grand_mean
            + model_effects[model]
            + task_effects[task]
            + prompt_effects[prompt]
            + run_effects[run]
            + rng.normal(0.0, np.sqrt(residual_variance))
        )
        value = float(np.clip(value, 0.0, 1.0))
        rows.append(
            {
                "model": model,
                "task_id": task,
                "prompt_id": prompt,
                "run_id": run,
                "temperature": temp,
                "metric_name": metric_name,
                "metric_value": value,
            }
        )

    true_components = {
        "model": model_variance,
        "task_id": task_variance,
        "prompt_id": prompt_variance,
        "run_id": run_variance,
        "temperature": temperature_variance,
        "residual": residual_variance,
        "total": (
            model_variance
            + task_variance
            + prompt_variance
            + run_variance
            + temperature_variance
            + residual_variance
        ),
    }
    return pd.DataFrame(rows), true_components
