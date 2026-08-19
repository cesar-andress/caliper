"""Paired statistical tests for v1.1 arms vs freeze (run only when both arms complete)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .constants import ANALYSIS_DIR, BOOTSTRAP_REPS, RANDOM_SEED


def mcnemar_from_binary(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left = left.astype(int)
    right = right.astype(int)
    b = int(((left == 1) & (right == 0)).sum())  # left success, right fail
    c = int(((left == 0) & (right == 1)).sum())  # left fail, right success
    # exact binomial McNemar mid-p approximation via chi2 with continuity
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "note": "no discordant pairs"}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(1 - stats.chi2.cdf(chi2, df=1))
    return {"b": b, "c": c, "statistic": float(chi2), "p_value": p}


def paired_bootstrap_ci(delta: np.ndarray, reps: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    delta = np.asarray(delta, dtype=float)
    n = len(delta)
    boots = np.empty(reps, dtype=float)
    for i in range(reps):
        boots[i] = delta[rng.integers(0, n, size=n)].mean()
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "mean": float(delta.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "std": float(delta.std(ddof=1)) if n > 1 else 0.0,
    }


def bland_altman(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mean = (left + right) / 2.0
    diff = right - left
    md = float(diff.mean())
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    return {
        "mean_difference": md,
        "loa_low": md - 1.96 * sd,
        "loa_high": md + 1.96 * sd,
        "mean_of_means": float(mean.mean()),
        "sd_difference": sd,
    }


def kendall_tau_task_ranks(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    l = left.groupby("task_id")["pass_at_1"].mean()
    r = right.groupby("task_id")["pass_at_1"].mean()
    common = l.index.intersection(r.index)
    if len(common) < 3:
        return {"tau": None, "p_value": None, "n": int(len(common))}
    tau, p = stats.kendalltau(l.loc[common], r.loc[common])
    return {"tau": float(tau), "p_value": float(p), "n": int(len(common))}


def spearman_task(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    l = left.groupby("task_id")["pass_at_1"].mean()
    r = right.groupby("task_id")["pass_at_1"].mean()
    common = l.index.intersection(r.index)
    if len(common) < 3:
        return {"rho": None, "p_value": None, "n": int(len(common))}
    rho, p = stats.spearmanr(l.loc[common], r.loc[common])
    return {"rho": float(rho), "p_value": float(p), "n": int(len(common))}


def analyze_pair(
    paired: pd.DataFrame,
    left: str,
    right: str,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
) -> dict[str, Any]:
    pl = paired[f"pass_at_1_{left}"].to_numpy(dtype=float)
    pr = paired[f"pass_at_1_{right}"].to_numpy(dtype=float)
    # binarize pass for McNemar
    mcn = mcnemar_from_binary((pl >= 0.5).astype(int), (pr >= 0.5).astype(int))
    delta = pr - pl
    boot = paired_bootstrap_ci(delta, BOOTSTRAP_REPS, RANDOM_SEED)
    # effect size: standardized mean difference on cell deltas
    d = float(delta.mean() / delta.std(ddof=1)) if len(delta) > 1 and delta.std(ddof=1) > 0 else 0.0
    ba = bland_altman(pl, pr)
    return {
        "mcnemar_pass": mcn,
        "paired_bootstrap_pass_delta": boot,
        "cohens_d_pass_delta": d,
        "bland_altman_pass": ba,
        "kendall_task": kendall_tau_task_ranks(left_df, right_df),
        "spearman_task": spearman_task(left_df, right_df),
        "n_paired": int(len(paired)),
    }


def run_statistical_comparisons(
    freeze: pd.DataFrame,
    arm_a: pd.DataFrame,
    arm_b: pd.DataFrame,
    paired_a: pd.DataFrame,
    paired_b: pd.DataFrame,
    paired_ab: pd.DataFrame,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    out_dir = out_dir or (ANALYSIS_DIR / "statistics")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "A_vs_freeze": analyze_pair(paired_a, "freeze", "A", freeze, arm_a),
        "B_vs_freeze": analyze_pair(paired_b, "freeze", "B", freeze, arm_b),
        "A_vs_B": analyze_pair(paired_ab, "A", "B", arm_a, arm_b),
    }

    # Also McNemar on emptiness
    for name, paired, left, right in [
        ("A_vs_freeze", paired_a, "freeze", "A"),
        ("B_vs_freeze", paired_b, "freeze", "B"),
        ("A_vs_B", paired_ab, "A", "B"),
    ]:
        el = paired[f"empty_{left}"].astype(int).to_numpy()
        er = paired[f"empty_{right}"].astype(int).to_numpy()
        results[name]["mcnemar_empty"] = mcnemar_from_binary(el, er)

    (out_dir / "statistical_comparisons.json").write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )
    rows = []
    for name, block in results.items():
        rows.append(
            {
                "comparison": name,
                "n_paired": block["n_paired"],
                "pass_delta_mean": block["paired_bootstrap_pass_delta"]["mean"],
                "pass_delta_ci_low": block["paired_bootstrap_pass_delta"]["ci_low"],
                "pass_delta_ci_high": block["paired_bootstrap_pass_delta"]["ci_high"],
                "cohens_d": block["cohens_d_pass_delta"],
                "mcnemar_pass_p": block["mcnemar_pass"]["p_value"],
                "mcnemar_empty_p": block["mcnemar_empty"]["p_value"],
                "kendall_tau": block["kendall_task"]["tau"],
                "spearman_rho": block["spearman_task"]["rho"],
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "statistical_comparisons_summary.csv", index=False)
    return results
