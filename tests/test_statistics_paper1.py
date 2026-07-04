"""Tests for Paper 1 statistical analysis module."""

from __future__ import annotations

import pandas as pd
import pytest

from caliper.statistics.bootstrap import bootstrap_ci, bootstrap_ci_by_factor
from caliper.statistics.descriptive import descriptive_by_factor
from caliper.statistics.gtheory import (
    compute_g_coefficient,
    estimate_g_variance_components,
    simulate_d_study,
    simulate_d_study_grid,
)
from caliper.statistics.mixed_effects import fit_mixed_model
from caliper.statistics.power_sim import simulate_power, simulate_power_grid
from caliper.statistics.prepare import prepare_results_table
from caliper.statistics.synthetic import generate_synthetic_results
from caliper.statistics.variance import decompose_variance, estimate_variance_components


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, dict[str, float]]:
    return generate_synthetic_results(
        n_models=3,
        n_tasks=8,
        n_prompts=3,
        n_runs=4,
        model_variance=0.05,
        task_variance=0.10,
        prompt_variance=0.02,
        run_variance=0.01,
        residual_variance=0.05,
        seed=7,
    )


class TestPrepareResults:
    def test_maps_aliases(self) -> None:
        df = pd.DataFrame(
            {
                "model_id": ["a"],
                "task": ["t1"],
                "prompt_variant_id": ["p1"],
                "run_index": [0],
                "score": [0.9],
            }
        )
        out = prepare_results_table(df)
        assert "model" in out.columns
        assert "task_id" in out.columns
        assert "metric_value" in out.columns

    def test_filters_metric(self) -> None:
        df = pd.DataFrame(
            {
                "model": ["a", "a"],
                "task_id": ["t1", "t1"],
                "metric_name": ["exact_match", "rouge_l"],
                "metric_value": [0.9, 0.5],
            }
        )
        out = prepare_results_table(df, metric_name="exact_match")
        assert len(out) == 1


class TestDescriptive:
    def test_by_model(self, synthetic_data: tuple[pd.DataFrame, dict[str, float]]) -> None:
        df, _ = synthetic_data
        summary = descriptive_by_factor(df, "model")
        assert len(summary) == 3
        assert "mean" in summary.columns
        assert summary["count"].iloc[0] > 0


class TestBootstrap:
    def test_ci_contains_mean(self) -> None:
        values = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        result = bootstrap_ci(values, n_bootstrap=500, seed=0)
        assert result.lower <= result.statistic <= result.upper

    def test_by_factor(self, synthetic_data: tuple[pd.DataFrame, dict[str, float]]) -> None:
        df, _ = synthetic_data
        result = bootstrap_ci_by_factor(df, "model", n_bootstrap=200, seed=1)
        assert len(result) == 3
        assert "ci_lower" in result.columns


class TestVarianceDecomposition:
    def test_recovers_components_approximately(
        self,
        synthetic_data: tuple[pd.DataFrame, dict[str, float]],
    ) -> None:
        df, true = synthetic_data
        estimated = estimate_variance_components(
            df,
            ["model", "task_id", "prompt_id", "run_id"],
        )

        # Sequential ANOVA is approximate; allow generous tolerance.
        assert abs(estimated["task_id"] - true["task_id"]) < 0.08
        assert abs(estimated["model"] - true["model"]) < 0.08
        assert abs(estimated["residual"] - true["residual"]) < 0.08
        assert estimated["total"] > 0

    def test_legacy_columns(self) -> None:
        df = pd.DataFrame(
            {
                "model": ["a", "b"],
                "task": ["t1", "t1"],
                "prompt_id": ["p1", "p1"],
                "run_index": [0, 0],
                "score": [0.8, 0.9],
            }
        )
        result = decompose_variance(df)
        assert result.n_observations == 2


class TestMixedEffects:
    def test_fit_returns_result(self, synthetic_data: tuple[pd.DataFrame, dict[str, float]]) -> None:
        df, _ = synthetic_data
        result = fit_mixed_model(df, group_col="task_id")
        assert result.method in ("statsmodels_mixedlm", "sequential_anova_fallback")
        assert result.variance_components

    def test_fallback_on_tiny_data(self) -> None:
        df = pd.DataFrame(
            {
                "model": ["a"],
                "task_id": ["t1"],
                "metric_value": [0.5],
            }
        )
        result = fit_mixed_model(df, group_col="task_id")
        assert result.method == "sequential_anova_fallback"


class TestGTheory:
    def test_g_coefficient_range(self, synthetic_data: tuple[pd.DataFrame, dict[str, float]]) -> None:
        df, _ = synthetic_data
        components = estimate_g_variance_components(df).components
        g = compute_g_coefficient(components, ["task_id", "prompt_id"])
        assert 0.0 <= g <= 1.0

    def test_d_study_improves_with_more_tasks(
        self,
        synthetic_data: tuple[pd.DataFrame, dict[str, float]],
    ) -> None:
        df, _ = synthetic_data
        components = estimate_g_variance_components(df).components
        small = simulate_d_study(components, {"task_id": 1, "prompt_id": 1, "run_id": 1})
        large = simulate_d_study(components, {"task_id": 10, "prompt_id": 3, "run_id": 5})
        assert large.expected_observed_variance <= small.expected_observed_variance

    def test_d_study_grid_shape(self, synthetic_data: tuple[pd.DataFrame, dict[str, float]]) -> None:
        df, _ = synthetic_data
        components = estimate_g_variance_components(df).components
        grid = simulate_d_study_grid(
            components,
            task_counts=[1, 3],
            prompt_counts=[1, 2],
            run_counts=[1, 2],
        )
        assert len(grid) == 8


class TestPowerSimulation:
    def test_power_increases_with_effect_size(
        self,
        synthetic_data: tuple[pd.DataFrame, dict[str, float]],
    ) -> None:
        _, components = synthetic_data
        low = simulate_power(components, effect_size=0.01, n_simulations=200, seed=0)
        high = simulate_power(components, effect_size=0.20, n_simulations=200, seed=0)
        assert high.power >= low.power

    def test_power_grid(self, synthetic_data: tuple[pd.DataFrame, dict[str, float]]) -> None:
        _, components = synthetic_data
        grid = simulate_power_grid(
            components,
            effect_size=0.05,
            task_counts=[3, 5],
            prompt_counts=[1, 2],
            run_counts=[1, 3],
            n_simulations=100,
            seed=1,
        )
        assert len(grid) == 8
        assert (grid["power"] >= 0).all()
        assert (grid["power"] <= 1).all()

    def test_more_runs_increases_power(
        self,
        synthetic_data: tuple[pd.DataFrame, dict[str, float]],
    ) -> None:
        _, components = synthetic_data
        few = simulate_power(
            components, effect_size=0.08, n_tasks=5, n_prompts=2, n_runs=1, n_simulations=300, seed=2
        )
        many = simulate_power(
            components, effect_size=0.08, n_tasks=5, n_prompts=2, n_runs=5, n_simulations=300, seed=2
        )
        assert many.power >= few.power
