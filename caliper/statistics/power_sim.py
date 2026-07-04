"""Power analysis simulation for Paper 1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from caliper.statistics.power import PowerAnalysisResult, compute_power


@dataclass(frozen=True)
class PowerSimulationResult:
    """Result of a Monte Carlo power simulation."""

    effect_size: float
    alpha: float
    n_tasks: int
    n_prompts: int
    n_runs: int
    n_simulations: int
    power: float
    test: str

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "effect_size": self.effect_size,
            "alpha": self.alpha,
            "n_tasks": self.n_tasks,
            "n_prompts": self.n_prompts,
            "n_runs": self.n_runs,
            "n_simulations": self.n_simulations,
            "power": self.power,
            "test": self.test,
        }


def _simulate_two_model_means(
    *,
    effect_size: float,
    n_tasks: int,
    n_prompts: int,
    n_runs: int,
    variance_components: dict[str, float],
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Simulate mean scores for two models under a crossed design."""
    task_sd = np.sqrt(max(variance_components.get("task_id", 0.0), 0.0))
    prompt_sd = np.sqrt(max(variance_components.get("prompt_id", 0.0), 0.0))
    run_sd = np.sqrt(max(variance_components.get("run_id", 0.0), 0.0))
    residual_sd = np.sqrt(max(variance_components.get("residual", 0.0), 0.1))

    scores_a: list[float] = []
    scores_b: list[float] = []
    for _task in range(n_tasks):
        task_effect = rng.normal(0.0, task_sd)
        for _prompt in range(n_prompts):
            prompt_effect = rng.normal(0.0, prompt_sd)
            for _run in range(n_runs):
                run_effect = rng.normal(0.0, run_sd)
                error_a = rng.normal(0.0, residual_sd)
                error_b = rng.normal(0.0, residual_sd)
                scores_a.append(task_effect + prompt_effect + run_effect + error_a)
                scores_b.append(
                    task_effect + prompt_effect + run_effect + error_b + effect_size
                )
    return float(np.mean(scores_a)), float(np.mean(scores_b))


def simulate_power(
    variance_components: dict[str, float],
    *,
    effect_size: float,
    n_tasks: int = 5,
    n_prompts: int = 2,
    n_runs: int = 3,
    alpha: float = 0.05,
    n_simulations: int = 500,
    seed: int = 42,
) -> PowerSimulationResult:
    """Estimate power via Monte Carlo simulation of a two-model comparison.

    Each simulation generates crossed task × prompt × run measurements for two
    models differing by ``effect_size`` in mean score, then applies a two-sample
    t-test on cell means.
    """
    rng = np.random.default_rng(seed)
    n_cells = n_tasks * n_prompts * n_runs
    if n_cells < 2:
        return PowerSimulationResult(
            effect_size=effect_size,
            alpha=alpha,
            n_tasks=n_tasks,
            n_prompts=n_prompts,
            n_runs=n_runs,
            n_simulations=n_simulations,
            power=0.0,
            test="simulated_t_test",
        )

    rejections = 0
    for _ in range(n_simulations):
        mean_a, mean_b = _simulate_two_model_means(
            effect_size=effect_size,
            n_tasks=n_tasks,
            n_prompts=n_prompts,
            n_runs=n_runs,
            variance_components=variance_components,
            rng=rng,
        )
        # Approximate t-test using aggregated means and residual variance.
        residual_var = max(variance_components.get("residual", 0.1), 1e-6)
        se = np.sqrt(2 * residual_var / n_cells)
        if se == 0:
            continue
        t_stat = (mean_b - mean_a) / se
        df = 2 * n_cells - 2
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))
        if p_value < alpha:
            rejections += 1

    power = rejections / n_simulations
    return PowerSimulationResult(
        effect_size=effect_size,
        alpha=alpha,
        n_tasks=n_tasks,
        n_prompts=n_prompts,
        n_runs=n_runs,
        n_simulations=n_simulations,
        power=float(power),
        test="simulated_t_test",
    )


def simulate_power_grid(
    variance_components: dict[str, float],
    *,
    effect_size: float,
    task_counts: list[int],
    prompt_counts: list[int],
    run_counts: list[int],
    alpha: float = 0.05,
    n_simulations: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate power over a grid of task, prompt, and run counts."""
    rows: list[dict[str, float | int]] = []
    offset = 0
    for n_tasks in task_counts:
        for n_prompts in prompt_counts:
            for n_runs in run_counts:
                result = simulate_power(
                    variance_components,
                    effect_size=effect_size,
                    n_tasks=n_tasks,
                    n_prompts=n_prompts,
                    n_runs=n_runs,
                    alpha=alpha,
                    n_simulations=n_simulations,
                    seed=seed + offset,
                )
                offset += 1
                rows.append(result.as_dict())  # type: ignore[arg-type]
    return pd.DataFrame(rows)


__all__ = [
    "PowerAnalysisResult",
    "PowerSimulationResult",
    "compute_power",
    "simulate_power",
    "simulate_power_grid",
]
