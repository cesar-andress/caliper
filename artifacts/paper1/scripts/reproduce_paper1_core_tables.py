#!/usr/bin/env python3
"""Reproduce core Paper 1 descriptive tables from the frozen statistical dataset.

No model inference. Writes CSV summaries under ./repro_out/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "frozen" / "statistical_dataset.parquet"
OUT = Path(__file__).resolve().parent / "repro_out"
NONCOMPLIANT = "qwen3_32b"


def type1(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    y = df["metric_value"].astype(float).to_numpy()
    grand = y.mean()
    ss_tot = float(np.sum((y - grand) ** 2))
    residual = y.copy()
    rows = []
    for facet in order:
        gmean = df.assign(_r=residual).groupby(facet, observed=True)["_r"].transform("mean").to_numpy()
        ss = float(np.sum((gmean - residual.mean()) ** 2))
        rows.append({"component": facet, "pct": 100.0 * ss / ss_tot})
        residual = residual - (gmean - residual.mean())
    ss_res = float(np.sum((residual - residual.mean()) ** 2))
    rows.append({"component": "residual", "pct": 100.0 * ss_res / ss_tot})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    df = df[df["metric_name"] == "pass_at_1"].copy()
    assert len(df) == 39360

    compliant = df[df["model"] != NONCOMPLIANT].copy()
    order = ["model", "task_id", "prompt_id", "run_index", "temperature"]

    type1(df, order).to_csv(OUT / "type1_full6.csv", index=False)
    type1(compliant, order).to_csv(OUT / "type1_compliant5.csv", index=False)

    # extraction diagnostics
    preds = df["prediction"].fillna("")
    empty = preds.str.strip().eq("")
    fence = preds.str.contains(r"```", regex=True)
    ext = (
        df.assign(empty=empty, fence=fence)
        .groupby("model", observed=True)
        .agg(
            n=("metric_value", "size"),
            mean_pass=("metric_value", "mean"),
            empty_rate=("empty", "mean"),
            fence_rate=("fence", "mean"),
        )
        .reset_index()
    )
    ext.to_csv(OUT / "extraction_by_model.csv", index=False)

    print(f"Wrote reproduction CSVs to {OUT}")
    print("Compliant Type-I:")
    print(type1(compliant, order).to_string(index=False))


if __name__ == "__main__":
    main()
