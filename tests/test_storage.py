"""Tests for result storage helpers."""

from pathlib import Path

import pandas as pd
import pytest

from caliper.storage.formats import read_results, write_manifest, write_results


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "model": ["a", "b"],
        "score": [0.9, 0.8],
        "task": ["t1", "t1"],
    })


class TestWriteReadResults:
    @pytest.mark.parametrize("fmt,ext", [("parquet", ".parquet"), ("csv", ".csv"), ("jsonl", ".jsonl")])
    def test_round_trip(self, tmp_path: Path, sample_df: pd.DataFrame, fmt: str, ext: str) -> None:
        path = tmp_path / f"results{ext}"
        write_results(sample_df, path, fmt=fmt)  # type: ignore[arg-type]
        loaded = read_results(path)
        assert len(loaded) == 2
        assert list(loaded.columns) == list(sample_df.columns)

    def test_unsupported_format(self, tmp_path: Path, sample_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unsupported format"):
            write_results(sample_df, tmp_path / "out.xyz", fmt="xyz")  # type: ignore[arg-type]


class TestWriteManifest:
    def test_writes_json(self, tmp_path: Path) -> None:
        manifest = {"run_id": "abc123", "status": "completed"}
        path = write_manifest(manifest, tmp_path / "manifest.json")
        assert path.exists()
        assert "abc123" in path.read_text()
