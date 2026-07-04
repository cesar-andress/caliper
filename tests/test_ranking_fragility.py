"""Tests for Paper 2 ranking fragility module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from caliper.ranking.analysis import run_ranking_fragility_analysis
from caliper.ranking.aggregate import rank_models
from caliper.ranking.bootstrap import bootstrap_rankings
from caliper.ranking.fragility import compute_ranking_fragility
from caliper.ranking.metrics import (
    kendall_tau_between_rankings,
    rank_probability_matrix,
    ranking_fragility_index,
)
from caliper.ranking.synthetic import generate_stable_ranking_data, generate_unstable_ranking_data


class TestAggregate:
    def test_rank_models_order(self) -> None:
        scores = pd.Series({"a": 0.9, "b": 0.7, "c": 0.8})
        ranks = rank_models(scores)
        assert ranks["a"] == 1.0
        assert ranks["b"] == 3.0
        assert ranks["c"] == 2.0


class TestMetrics:
    def test_kendall_tau_identical(self) -> None:
        ranks = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
        assert kendall_tau_between_rankings(ranks, ranks) == pytest.approx(1.0)

    def test_fragility_index_stable(self) -> None:
        assert ranking_fragility_index([1.0, 1.0, 1.0]) == pytest.approx(0.0)

    def test_fragility_index_unstable(self) -> None:
        assert ranking_fragility_index([-1.0, -1.0]) == pytest.approx(1.0)

    def test_rank_probability_sums_to_one(self) -> None:
        samples = pd.DataFrame(
            {
                "model": ["a", "a", "b", "b"],
                "rank": [1, 2, 2, 1],
                "iteration": [0, 1, 0, 1],
            }
        )
        probs = rank_probability_matrix(samples, n_ranks=2)
        assert probs.loc["a"].sum() == pytest.approx(1.0)
        assert probs.loc["b"].sum() == pytest.approx(1.0)


class TestBootstrap:
    def test_bootstrap_produces_samples(self) -> None:
        df = generate_stable_ranking_data(n_models=3, n_tasks=10, seed=0)
        samples, baseline, taus = bootstrap_rankings(df, "task_id", n_bootstrap=50, seed=0)
        assert len(samples) == 50 * 3
        assert len(baseline) == 3
        assert len(taus) == 50
        assert all(-1.0 <= t <= 1.0 for t in taus)


class TestStableVsUnstable:
    def test_stable_rankings_high_tau(self) -> None:
        df = generate_stable_ranking_data(n_models=4, n_tasks=30, seed=0)
        _, _, taus = bootstrap_rankings(df, "task_id", n_bootstrap=100, seed=0)
        assert float(pd.Series(taus).mean()) > 0.95

    def test_unstable_rankings_lower_tau(self) -> None:
        stable = generate_stable_ranking_data(n_models=4, n_tasks=30, seed=0)
        unstable = generate_unstable_ranking_data(n_models=4, n_tasks=30, seed=1)

        _, _, stable_taus = bootstrap_rankings(stable, "task_id", n_bootstrap=100, seed=0)
        _, _, unstable_taus = bootstrap_rankings(unstable, "task_id", n_bootstrap=100, seed=0)

        assert float(pd.Series(unstable_taus).mean()) < float(pd.Series(stable_taus).mean())

    def test_stable_lower_fragility_index(self, tmp_path: Path) -> None:
        stable = generate_stable_ranking_data(n_models=4, n_tasks=25, seed=0)
        unstable = generate_unstable_ranking_data(n_models=4, n_tasks=25, seed=1)

        stable_out = run_ranking_fragility_analysis(
            stable, n_bootstrap=80, seed=0, output_dir=tmp_path / "stable"
        )
        unstable_out = run_ranking_fragility_analysis(
            unstable, n_bootstrap=80, seed=0, output_dir=tmp_path / "unstable"
        )

        stable_idx = stable_out.summary.loc[
            stable_out.summary["bootstrap_type"] == "overall", "fragility_index"
        ].iloc[0]
        unstable_idx = unstable_out.summary.loc[
            unstable_out.summary["bootstrap_type"] == "overall", "fragility_index"
        ].iloc[0]
        assert unstable_idx > stable_idx


class TestAnalysisOutputs:
    def test_writes_outputs(self, tmp_path: Path) -> None:
        df = generate_stable_ranking_data(n_models=3, n_tasks=15, seed=0)
        outputs = run_ranking_fragility_analysis(
            df,
            n_bootstrap=50,
            seed=0,
            output_dir=tmp_path / "out",
            reports_dir=tmp_path / "reports",
        )
        assert (tmp_path / "out" / "ranking_fragility_summary.csv").exists()
        assert (tmp_path / "out" / "bootstrap_samples.parquet").exists()
        assert (tmp_path / "out" / "bootstrap_samples.jsonl").exists()
        assert (tmp_path / "out" / "rank_probabilities.csv").exists()
        assert (tmp_path / "out" / "pairwise_reversals.csv").exists()
        assert (tmp_path / "reports" / "kendall_tau_distribution.png").exists()
        assert len(outputs.summary) >= 4

    def test_pairwise_reversals_bounded(self, tmp_path: Path) -> None:
        df = generate_unstable_ranking_data(n_models=4, n_tasks=20, seed=1)
        outputs = run_ranking_fragility_analysis(
            df, n_bootstrap=60, seed=0, output_dir=tmp_path / "out"
        )
        assert (outputs.pairwise_reversals["reversal_probability"] >= 0).all()
        assert (outputs.pairwise_reversals["reversal_probability"] <= 1).all()


class TestLegacyAPI:
    def test_compute_ranking_fragility(self) -> None:
        df = generate_stable_ranking_data(n_models=3, n_tasks=5, seed=0)
        df_legacy = df.rename(columns={"metric_value": "score"})
        result = compute_ranking_fragility(df_legacy, n_perturbations=50, seed=0)
        assert result.n_models == 3
        assert 0 <= result.fragility_rate <= 1
