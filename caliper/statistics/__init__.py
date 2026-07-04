"""Statistical analysis: variance decomposition and power (Paper 1)."""

from caliper.statistics.bootstrap import BootstrapResult, bootstrap_ci, bootstrap_ci_by_factor
from caliper.statistics.descriptive import descriptive_all_factors, descriptive_by_factor
from caliper.statistics.gtheory import (
    DStudyResult,
    GStudyResult,
    compute_g_coefficient,
    compute_phi_coefficient,
    estimate_g_variance_components,
    expected_observed_score_variance,
    simulate_d_study,
    simulate_d_study_grid,
)
from caliper.statistics.mixed_effects import MixedModelResult, fit_mixed_model
from caliper.statistics.power import PowerAnalysisResult, compute_power
from caliper.statistics.power_sim import PowerSimulationResult, simulate_power, simulate_power_grid
from caliper.statistics.prepare import prepare_results_table
from caliper.statistics.synthetic import generate_synthetic_results
from caliper.statistics.variance import (
    VarianceComponents,
    decompose_variance,
    estimate_variance_components,
)

__all__ = [
    "BootstrapResult",
    "DStudyResult",
    "GStudyResult",
    "MixedModelResult",
    "PowerAnalysisResult",
    "PowerSimulationResult",
    "VarianceComponents",
    "bootstrap_ci",
    "bootstrap_ci_by_factor",
    "compute_g_coefficient",
    "compute_phi_coefficient",
    "compute_power",
    "decompose_variance",
    "descriptive_all_factors",
    "descriptive_by_factor",
    "estimate_g_variance_components",
    "estimate_variance_components",
    "expected_observed_score_variance",
    "fit_mixed_model",
    "generate_synthetic_results",
    "prepare_results_table",
    "simulate_d_study",
    "simulate_d_study_grid",
    "simulate_power",
    "simulate_power_grid",
]
