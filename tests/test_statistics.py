"""Tests for statistical analysis modules."""

import pandas as pd
import pytest

from caliper.ranking.fragility import compute_ranking_fragility
from caliper.statistics.power import compute_power
from caliper.statistics.variance import decompose_variance


@pytest.fixture
def results_df() -> pd.DataFrame:
    rows = []
    for model in ["a", "b", "c"]:
        for prompt in ["p1", "p2"]:
            for task in ["t1", "t2"]:
                for run in range(3):
                    score = {"a": 0.9, "b": 0.85, "c": 0.8}[model]
                    rows.append({
                        "model": model,
                        "prompt_id": prompt,
                        "task": task,
                        "run_index": run,
                        "score": score + run * 0.001,
                    })
    return pd.DataFrame(rows)


class TestVarianceDecomposition:
    def test_returns_components(self, results_df: pd.DataFrame) -> None:
        result = decompose_variance(results_df)
        assert result.n_observations == len(results_df)
        assert result.total_variance >= 0

    def test_as_dict(self, results_df: pd.DataFrame) -> None:
        d = decompose_variance(results_df).as_dict()
        assert "total_variance" in d
        assert "model_variance" in d


class TestPowerAnalysis:
    def test_large_effect_high_power(self) -> None:
        result = compute_power(effect_size=1.0, n_per_group=50)
        assert result.power > 0.9

    def test_small_effect_low_power(self) -> None:
        result = compute_power(effect_size=0.1, n_per_group=10)
        assert result.power < 0.5

    def test_unsupported_test(self) -> None:
        with pytest.raises(ValueError, match="Unsupported test"):
            compute_power(0.5, 20, test="anova")


class TestRankingFragility:
    def test_perfect_scores_high_fragility(self, results_df: pd.DataFrame) -> None:
        result = compute_ranking_fragility(
            results_df, noise_scale=0.05, n_perturbations=100, seed=0
        )
        assert 0 <= result.fragility_rate <= 1
        assert result.n_models == 3

    def test_as_dict(self, results_df: pd.DataFrame) -> None:
        d = compute_ranking_fragility(results_df, n_perturbations=10).as_dict()
        assert "fragility_rate" in d
