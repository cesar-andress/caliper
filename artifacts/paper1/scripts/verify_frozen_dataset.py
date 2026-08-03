#!/usr/bin/env python3
"""Verify Paper 1 frozen statistical dataset integrity (no inference)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "frozen" / "statistical_dataset.parquet"
EXPECTED_SHA256 = "95209fff2f742d59b52aa5cf5616f1395ef7fc01fd87fe025c485577bca0d1c9"
EXPECTED_N = 39360
EXPECTED_METRIC = "pass_at_1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not FROZEN.is_file():
        print(f"FAIL: missing {FROZEN}", file=sys.stderr)
        return 1
    digest = sha256(FROZEN)
    if digest != EXPECTED_SHA256:
        print(f"FAIL: sha256 mismatch\n expected {EXPECTED_SHA256}\n got      {digest}", file=sys.stderr)
        return 1
    df = pd.read_parquet(FROZEN)
    if "metric_name" in df.columns:
        df = df[df["metric_name"] == EXPECTED_METRIC]
    n = len(df)
    n_cells = df["cell_id"].nunique() if "cell_id" in df.columns else n
    if n != EXPECTED_N or n_cells != EXPECTED_N:
        print(f"FAIL: row/cell count {n}/{n_cells}, expected {EXPECTED_N}", file=sys.stderr)
        return 1
    print("OK: statistical_dataset.parquet")
    print(f"  sha256: {digest}")
    print(f"  rows:   {n}")
    print(f"  cells:  {n_cells}")
    print(f"  mean pass@1: {df['metric_value'].mean():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
