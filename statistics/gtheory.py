"""Generalizability theory helpers for Paper 1."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from caliper.statistics.variance import estimate_variance_components


@dataclass(frozen=True)
class GStudyResult:
    """Generalizability study variance component estimates."""

    components: dict[str, float]
    n_observations: int
    facets: list[str]

    def as_dict(self) -> dict[str, float | int | list[str]]:
        return {
            "components": self.components,
            "n_observations": self.n_observations,
            "facets": self.facets,
        }


@dataclass(frozen=True)
class DStudyResult:
    """Decision (D) study simulation for a proposed measurement design."""

    design: dict[str, int]
    g_coefficient: float
    phi_coefficient: float
    expected_observed_variance: float
    variance_components: dict[str, float]

    def as_dict(self) -> dict[str, float | dict[str, int] | dict[str, float]]:
        return {
            "design": self.design,
            "g_coefficient": self.g_coefficient,
            "phi_coefficient": self.phi_coefficient,
            "expected_observed_variance": self.expected_observed_variance,
            "variance_components": self.variance_components,
        }


def estimate_g_variance_components(
    df: pd.DataFrame,
    facets: list[str] | None = None,
    *,
    value_col: str = "metric_value",
) -> GStudyResult:
    """Estimate variance components for a G-study."""
    if facets is None:
        facets = [c for c in ("model", "task_id", "prompt_id", "run_id", "temperature") if c in df.columns]

    components = estimate_variance_components(df, facets, value_col=value_col)
    return GStudyResult(
        components=components,
        n_observations=len(df),
        facets=facets,
    )

def compute_g_coefficient(
    components: dict[str, float],
    universe_facets: list[str],
) -> float:
    """Compute the G coefficient (relative generalizability).

    G = var_universe / (var_universe + var_residual)

    where var_universe is the sum of variance components for facets in the
    universe of generalization.
    """
    universe_var = sum(components.get(facet, 0.0) for facet in universe_facets)
    residual = components.get("residual", 0.0)
    denom = universe_var + residual
    return float(universe_var / denom) if denom > 0 else 0.0


def compute_phi_coefficient(
    components: dict[str, float],
    universe_facets: list[str],
    *,
    n_replications: dict[str, int] | None = None,
) -> float:
    """Compute the Φ coefficient for absolute decisions.

    Uses universe variance relative to total expected error variance under
    the proposed replication counts.
    """
    n_replications = n_replications or {}
    universe_var = sum(components.get(facet, 0.0) for facet in universe_facets)
    error_var = components.get("residual", 0.0)
    for facet, n in n_replications.items():
        if facet in universe_facets and n > 1:
            error_var += components.get(facet, 0.0) / n
    denom = universe_var + error_var
    return float(universe_var / denom) if denom > 0 else 0.0


def expected_observed_score_variance(
    components: dict[str, float],
    design: dict[str, int],
) -> float:
    """Expected variance of observed mean scores under a D-study design.

    Each facet contribution is divided by its replication count in ``design``.
    """
    var = components.get("residual", 0.0)
    for facet, n in design.items():
        if facet == "residual":
            continue
        if n <= 0:
            continue
        var += components.get(facet, 0.0) / n
    return float(max(var, 0.0))


def simulate_d_study(
    components: dict[str, float],
    design: dict[str, int],
    *,
    universe_facets: list[str] | None = None,
) -> DStudyResult:
    """Simulate a D-study given G-study components and a proposed design."""
    if universe_facets is None:
        universe_facets = [f for f in design if f != "residual"]

    obs_var = expected_observed_score_variance(components, design)
    g = compute_g_coefficient(components, universe_facets)
    phi = compute_phi_coefficient(components, universe_facets, n_replications=design)

    return DStudyResult(
        design=design,
        g_coefficient=g,
        phi_coefficient=phi,
        expected_observed_variance=obs_var,
        variance_components=components,
    )


def simulate_d_study_grid(
    components: dict[str, float],
    *,
    task_counts: list[int],
    prompt_counts: list[int],
    run_counts: list[int],
    universe_facets: list[str] | None = None,
) -> pd.DataFrame:
    """Simulate a grid of D-studies varying tasks, prompts, and runs."""
    rows: list[dict[str, float | int]] = []
    for n_tasks in task_counts:
        for n_prompts in prompt_counts:
            for n_runs in run_counts:
                design = {
                    "task_id": n_tasks,
                    "prompt_id": n_prompts,
                    "run_id": n_runs,
                }
                result = simulate_d_study(components, design, universe_facets=universe_facets)
                rows.append(
                    {
                        "n_tasks": n_tasks,
                        "n_prompts": n_prompts,
                        "n_runs": n_runs,
                        "g_coefficient": result.g_coefficient,
                        "phi_coefficient": result.phi_coefficient,
                        "expected_observed_variance": result.expected_observed_variance,
                    }
                )
    return pd.DataFrame(rows)
