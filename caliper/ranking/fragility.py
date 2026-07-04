"""Ranking fragility metrics for LLM benchmark comparisons (Paper 2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from caliper.ranking.metrics import kendall_tau_between_rankings, ranking_fragility_index


@dataclass(frozen=True)
class RankingFragilityResult:
    """Summary of how fragile model rankings are under perturbation (legacy API)."""

    n_models: int
    n_perturbations: int
    rank_changes: int
    fragility_rate: float
    kendall_tau_mean: float
    kendall_tau_std: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n_models": self.n_models,
            "n_perturbations": self.n_perturbations,
            "rank_changes": self.rank_changes,
            "fragility_rate": self.fragility_rate,
            "kendall_tau_mean": self.kendall_tau_mean,
            "kendall_tau_std": self.kendall_tau_std,
        }


def compute_ranking_fragility(
    df: pd.DataFrame,
    *,
    score_col: str = "score",
    model_col: str = "model",
    noise_scale: float = 0.01,
    n_perturbations: int = 1000,
    seed: int = 42,
) -> RankingFragilityResult:
    """Measure ranking fragility by adding noise to aggregated scores.

    Legacy noise-perturbation API retained for backward compatibility.
    For bootstrap-based analysis use ``run_ranking_fragility_analysis``.
    """
    rng = np.random.default_rng(seed)
    mean_scores = df.groupby(model_col)[score_col].mean()
    models = mean_scores.index.tolist()
    baseline_rank = mean_scores.rank(ascending=False)
    n_models = len(models)

    rank_changes = 0
    taus: list[float] = []

    for _ in range(n_perturbations):
        noisy = mean_scores.values + rng.normal(0, noise_scale, size=n_models)
        perturbed_scores = pd.Series(noisy, index=mean_scores.index)
        perturbed_rank = perturbed_scores.rank(ascending=False)
        if not np.array_equal(
            baseline_rank.rank(method="first").values,
            perturbed_rank.rank(method="first").values,
        ):
            rank_changes += 1
        taus.append(kendall_tau_between_rankings(baseline_rank, perturbed_rank))

    return RankingFragilityResult(
        n_models=n_models,
        n_perturbations=n_perturbations,
        rank_changes=rank_changes,
        fragility_rate=rank_changes / n_perturbations,
        kendall_tau_mean=float(np.mean(taus)),
        kendall_tau_std=float(np.std(taus)),
    )
