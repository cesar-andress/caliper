"""Heterogeneity of correction effects across tasks, prompts, temperatures, runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import ANALYSIS_DIR, BOOTSTRAP_REPS, RANDOM_SEED


def _bootstrap_mean_ci(values: np.ndarray, reps: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    boots = []
    n = len(values)
    for _ in range(reps):
        sample = values[rng.integers(0, n, size=n)]
        boots.append(float(sample.mean()))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"mean": float(values.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def _cohen_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions."""
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def task_difficulty_from_freeze(freeze: pd.DataFrame) -> pd.DataFrame:
    """Proxy difficulty: 1 - mean pass@1 on freeze nonempty cells; fallback all cells."""
    g = freeze.groupby("task_id", as_index=False).agg(
        freeze_pass=("pass_at_1", "mean"),
        freeze_empty=("empty", "mean"),
        n=("pass_at_1", "size"),
    )
    g["difficulty"] = 1.0 - g["freeze_pass"]
    g["difficulty_tertile"] = pd.qcut(
        g["difficulty"],
        q=3,
        labels=["easy", "medium", "hard"],
        duplicates="drop",
    )
    return g


def analyze_heterogeneity(
    freeze: pd.DataFrame,
    arm_a: pd.DataFrame,
    arm_b: pd.DataFrame,
    paired_a: pd.DataFrame,
    paired_b: pd.DataFrame,
    paired_ab: pd.DataFrame,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    out_dir = out_dir or (ANALYSIS_DIR / "heterogeneity")
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    difficulty = task_difficulty_from_freeze(freeze)
    difficulty.to_csv(tables / "task_difficulty_freeze.csv", index=False)

    results: dict[str, Any] = {"effects": {}, "by_factor": {}}

    def _effect_block(paired: pd.DataFrame, left: str, right: str, name: str) -> dict[str, Any]:
        pl = f"pass_at_1_{left}"
        pr = f"pass_at_1_{right}"
        el = f"empty_{left}"
        er = f"empty_{right}"
        delta = (paired[pr] - paired[pl]).to_numpy(dtype=float)
        empty_delta = (paired[er].astype(float) - paired[el].astype(float)).to_numpy(dtype=float)
        p_left = float(paired[pl].mean())
        p_right = float(paired[pr].mean())
        block = {
            "pass_delta": _bootstrap_mean_ci(delta, BOOTSTRAP_REPS, RANDOM_SEED),
            "empty_delta": _bootstrap_mean_ci(empty_delta, BOOTSTRAP_REPS, RANDOM_SEED + 1),
            "cohen_h_pass": _cohen_h(p_right, p_left),
            "n": int(len(paired)),
        }
        # Attach difficulty if task_id present
        task_col = "task_id" if "task_id" in paired.columns else None
        if task_col:
            tmp = paired.merge(difficulty[["task_id", "difficulty", "difficulty_tertile"]], on="task_id", how="left")
            tmp["pass_delta"] = tmp[pr] - tmp[pl]
            by_diff = (
                tmp.groupby("difficulty_tertile", observed=False)["pass_delta"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            by_diff.to_csv(tables / f"{name}_pass_delta_by_difficulty.csv", index=False)
            block["by_difficulty"] = by_diff.to_dict(orient="records")

            # Spearman: harder freeze tasks → larger pass gains?
            task_delta = tmp.groupby("task_id", as_index=False).agg(
                pass_delta=("pass_delta", "mean"),
                difficulty=("difficulty", "first"),
            )
            if len(task_delta) > 3:
                block["spearman_difficulty_vs_pass_delta"] = float(
                    task_delta["difficulty"].corr(task_delta["pass_delta"], method="spearman")
                )
            task_delta.to_csv(tables / f"{name}_task_pass_delta.csv", index=False)
        results["effects"][name] = block
        return block

    _effect_block(paired_a, "freeze", "A", "A_vs_freeze")
    _effect_block(paired_b, "freeze", "B", "B_vs_freeze")
    _effect_block(paired_ab, "A", "B", "A_vs_B")

    # Factor slices for A_vs_freeze (primary correction contrast)
    for factor in ("prompt_id", "temperature", "run_index"):
        if factor not in paired_a.columns:
            continue
        tmp = paired_a.copy()
        tmp["pass_delta"] = tmp["pass_at_1_A"] - tmp["pass_at_1_freeze"]
        tmp["empty_delta"] = tmp["empty_A"].astype(float) - tmp["empty_freeze"].astype(float)
        g = (
            tmp.groupby(factor, observed=False)
            .agg(
                n=("pass_delta", "size"),
                pass_delta_mean=("pass_delta", "mean"),
                empty_delta_mean=("empty_delta", "mean"),
                pass_A=("pass_at_1_A", "mean"),
                pass_freeze=("pass_at_1_freeze", "mean"),
            )
            .reset_index()
        )
        # bootstrap CI per group
        cis = []
        for key, sub in tmp.groupby(factor, observed=False):
            ci = _bootstrap_mean_ci(sub["pass_delta"].to_numpy(), BOOTSTRAP_REPS, RANDOM_SEED + hash(str(key)) % 10000)
            cis.append({factor: key, **{f"pass_delta_{k}": v for k, v in ci.items()}})
        ci_df = pd.DataFrame(cis)
        g = g.merge(ci_df, on=factor, how="left")
        g.to_csv(tables / f"A_vs_freeze_by_{factor}.csv", index=False)
        results["by_factor"][factor] = g.to_dict(orient="records")

    (out_dir / "heterogeneity_summary.json").write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )
    return results
