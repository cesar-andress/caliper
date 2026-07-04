"""Ranking fragility analysis orchestration (Paper 2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

from caliper.ranking.aggregate import aggregate_scores_by_model, rank_models
from caliper.ranking.bootstrap import bootstrap_all_facets
from caliper.ranking.metrics import (
    pairwise_reversal_probability,
    rank_probability_matrix,
    ranking_fragility_index,
)
from caliper.ranking.plots import (
    plot_baseline_rankings,
    plot_kendall_tau_distribution,
    plot_pairwise_reversal_heatmap,
    plot_rank_probability_heatmap,
)
from caliper.statistics.prepare import prepare_results_table
from caliper.storage.formats import read_results, write_results

logger = structlog.get_logger(__name__)


@dataclass
class RankingFragilityOutputs:
    """Paths and tables produced by a fragility analysis run."""

    summary: pd.DataFrame
    bootstrap_samples: pd.DataFrame
    rank_probabilities: pd.DataFrame
    pairwise_reversals: pd.DataFrame
    baseline_scores: pd.Series
    output_dir: Path
    plot_paths: dict[str, Path] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "n_models": len(self.baseline_scores),
            "n_bootstrap_rows": len(self.bootstrap_samples),
            "fragility_index_mean": float(self.summary["fragility_index"].mean()),
            "plot_paths": {k: str(v) for k, v in self.plot_paths.items()},
        }


def run_ranking_fragility_analysis(
    df: pd.DataFrame,
    *,
    metric_name: str | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
    output_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> RankingFragilityOutputs:
    """Run full ranking fragility analysis with bootstrap, metrics, and plots."""
    prepared = prepare_results_table(df, metric_name=metric_name)
    if prepared.empty:
        msg = "no rows to analyze after filtering"
        raise ValueError(msg)

    if output_dir is None:
        output_dir = Path("reports/ranking_fragility")
    output_dir.mkdir(parents=True, exist_ok=True)
    if reports_dir is None:
        reports_dir = output_dir / "plots"
    reports_dir.mkdir(parents=True, exist_ok=True)

    bootstrap_samples, baseline_scores, taus_by_facet = bootstrap_all_facets(
        prepared,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    baseline_ranks = rank_models(baseline_scores)
    summary_rows: list[dict[str, object]] = []

    for facet, taus in taus_by_facet.items():
        facet_samples = bootstrap_samples[bootstrap_samples["bootstrap_type"] == facet]
        rank_changes = facet_samples.drop_duplicates("iteration")["rank_changed"].mean()
        summary_rows.append(
            {
                "metric_name": metric_name or prepared["metric_name"].iloc[0],
                "bootstrap_type": facet,
                "n_models": len(baseline_scores),
                "n_bootstrap": n_bootstrap,
                "kendall_tau_mean": float(np.mean(taus)),
                "kendall_tau_std": float(np.std(taus)),
                "kendall_tau_min": float(np.min(taus)),
                "rank_change_rate": float(rank_changes),
                "fragility_index": ranking_fragility_index(taus),
            }
        )

    overall_taus = [t for taus in taus_by_facet.values() for t in taus]
    summary_rows.append(
        {
            "metric_name": metric_name or prepared["metric_name"].iloc[0],
            "bootstrap_type": "overall",
            "n_models": len(baseline_scores),
            "n_bootstrap": len(overall_taus),
            "kendall_tau_mean": float(np.mean(overall_taus)),
            "kendall_tau_std": float(np.std(overall_taus)),
            "kendall_tau_min": float(np.min(overall_taus)),
            "rank_change_rate": float(bootstrap_samples.drop_duplicates(["bootstrap_type", "iteration"])["rank_changed"].mean()),
            "fragility_index": ranking_fragility_index(overall_taus),
        }
    )

    summary = pd.DataFrame(summary_rows)
    rank_probs = rank_probability_matrix(bootstrap_samples)
    pairwise = pairwise_reversal_probability(bootstrap_samples, baseline_scores)

    # Persist outputs
    summary_path = output_dir / "ranking_fragility_summary.csv"
    summary.to_csv(summary_path, index=False)

    parquet_path = output_dir / "bootstrap_samples.parquet"
    jsonl_path = output_dir / "bootstrap_samples.jsonl"
    write_results(bootstrap_samples, parquet_path, fmt="parquet")
    bootstrap_samples.to_json(jsonl_path, orient="records", lines=True)

    rank_probs.to_csv(output_dir / "rank_probabilities.csv")
    pairwise.to_csv(output_dir / "pairwise_reversals.csv")

    baseline_df = pd.DataFrame(
        {
            "model": baseline_scores.index,
            "mean_score": baseline_scores.values,
            "baseline_rank": baseline_ranks.values,
        }
    )
    baseline_df.to_csv(output_dir / "baseline_rankings.csv", index=False)

    plot_paths = {
        "kendall_tau": plot_kendall_tau_distribution(
            bootstrap_samples, reports_dir / "kendall_tau_distribution.png"
        ),
        "rank_probability": plot_rank_probability_heatmap(
            rank_probs, reports_dir / "rank_probability_heatmap.png"
        ),
        "pairwise_reversal": plot_pairwise_reversal_heatmap(
            pairwise,
            list(baseline_scores.index),
            reports_dir / "pairwise_reversal_heatmap.png",
        ),
        "baseline_rankings": plot_baseline_rankings(
            baseline_scores, reports_dir / "baseline_rankings.png"
        ),
    }

    logger.info(
        "ranking_fragility.complete",
        n_models=len(baseline_scores),
        fragility_index=summary_rows[-1]["fragility_index"],
        output_dir=str(output_dir),
    )

    return RankingFragilityOutputs(
        summary=summary,
        bootstrap_samples=bootstrap_samples,
        rank_probabilities=rank_probs,
        pairwise_reversals=pairwise,
        baseline_scores=baseline_scores,
        output_dir=output_dir,
        plot_paths=plot_paths,
    )


def run_ranking_fragility_from_file(
    results_path: Path,
    *,
    metric_name: str | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
    output_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> RankingFragilityOutputs:
    """Load results from disk and run ranking fragility analysis."""
    df = read_results(results_path)
    if output_dir is None:
        output_dir = Path("reports") / results_path.stem
    if reports_dir is None:
        reports_dir = output_dir / "plots"
    return run_ranking_fragility_analysis(
        df,
        metric_name=metric_name,
        n_bootstrap=n_bootstrap,
        seed=seed,
        output_dir=output_dir,
        reports_dir=reports_dir,
    )
