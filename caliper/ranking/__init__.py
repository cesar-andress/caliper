"""Ranking fragility analysis (Paper 2)."""

from caliper.ranking.analysis import (
    RankingFragilityOutputs,
    run_ranking_fragility_analysis,
    run_ranking_fragility_from_file,
)
from caliper.ranking.aggregate import aggregate_scores_by_model, build_score_matrix, rank_models
from caliper.ranking.bootstrap import bootstrap_all_facets, bootstrap_rankings
from caliper.ranking.fragility import RankingFragilityResult, compute_ranking_fragility
from caliper.ranking.metrics import (
    kendall_tau_between_rankings,
    pairwise_reversal_probability,
    rank_probability_matrix,
    ranking_fragility_index,
)
from caliper.ranking.synthetic import generate_stable_ranking_data, generate_unstable_ranking_data

__all__ = [
    "RankingFragilityOutputs",
    "RankingFragilityResult",
    "aggregate_scores_by_model",
    "bootstrap_all_facets",
    "bootstrap_rankings",
    "build_score_matrix",
    "compute_ranking_fragility",
    "generate_stable_ranking_data",
    "generate_unstable_ranking_data",
    "kendall_tau_between_rankings",
    "pairwise_reversal_probability",
    "rank_models",
    "rank_probability_matrix",
    "ranking_fragility_index",
    "run_ranking_fragility_analysis",
    "run_ranking_fragility_from_file",
]
