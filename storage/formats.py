"""Concrete read/write helpers for Parquet, JSONL, and CSV."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pandas as pd

Format = Literal["parquet", "jsonl", "csv"]


def write_results(df: pd.DataFrame, path: Path, fmt: Format = "parquet") -> Path:
    """Write a results DataFrame to disk in the specified format."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "jsonl":
        df.to_json(path, orient="records", lines=True)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        msg = f"Unsupported format: {fmt}"
        raise ValueError(msg)

    return path


def read_results(path: Path) -> pd.DataFrame:
    """Read a results file, inferring format from extension."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, orient="records", lines=True)
    if suffix == ".csv":
        return pd.read_csv(path)
    msg = f"Cannot infer format from extension: {suffix}"
    raise ValueError(msg)


def write_manifest(manifest: dict, path: Path) -> Path:
    """Write a run manifest as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path
