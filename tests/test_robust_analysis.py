"""Tests for robust statistical analysis modules."""

from __future__ import annotations

import pandas as pd
import pytest

from caliper.statistics.convergence import analyze_convergence
from caliper.statistics.robust_analysis import (
    bootstrap_ranking_robustness,
    compare_methods,
    leave_one_out_sensitivity,
    run_anova,
)
from caliper.statistics.synthetic import generate_synthetic_results


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    df, _ = generate_synthetic_results(
        n_models=3,
        n_tasks=4,
        n_prompts=2,
        n_runs=2,
        seed=1,
    )
    return df


class TestRobustAnalysis:
    def test_type2_anova_returns_effects(self, synthetic_df: pd.DataFrame) -> None:
        table = run_anova(synthetic_df, anova_type=2)
        if table.empty:
            pytest.skip("synthetic design is rank-deficient for OLS ANOVA")
        assert "partial_eta_squared" in table.columns
        assert "p_value" in table.columns

    def test_type3_anova_returns_effects(self, synthetic_df: pd.DataFrame) -> None:
        table = run_anova(synthetic_df, anova_type=3)
        if table.empty:
            pytest.skip("synthetic design is rank-deficient for OLS ANOVA")
        assert "p_value" in table.columns

    def test_compare_methods(self, synthetic_df: pd.DataFrame) -> None:
        result = compare_methods(synthetic_df)
        assert isinstance(result.method_comparison, pd.DataFrame)

    def test_leave_one_out_sensitivity(self, synthetic_df: pd.DataFrame) -> None:
        table = leave_one_out_sensitivity(synthetic_df)
        assert not table.empty
        assert "kendall_tau_vs_full" in table.columns

    def test_bootstrap_ranking_robustness(self, synthetic_df: pd.DataFrame) -> None:
        summary, rank_ci, samples = bootstrap_ranking_robustness(
            synthetic_df,
            n_bootstrap=50,
            seed=0,
        )
        assert len(summary) == 2
        assert not rank_ci.empty
        assert not samples.empty


class TestConvergence:
    def test_convergence_subsets(self, synthetic_df: pd.DataFrame) -> None:
        table = analyze_convergence(
            synthetic_df,
            subset_sizes=(50, 100, len(synthetic_df)),
            seed=0,
        )
        assert len(table) == 3
        assert "kendall_tau_vs_full_ranking" in table.columns
