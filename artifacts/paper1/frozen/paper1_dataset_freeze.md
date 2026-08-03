# Paper 1 Dataset Freeze

| Field | Value |
|-------|-------|
| **Freeze timestamp (UTC)** | `2026-08-03T16:34:29.887883+00:00` |
| **Git commit** | `4792dc6004e84e4fa69d9781981330015cdcdf18` |
| **Git branch** | `main` |
| **Experiment ID** | `paper1_confirmatory_humaneval_full` |
| **Run ID** | `c1b58081b70e` |
| **Recovery run ID(s)** | `b3fb840135b7` |
| **Total expected cells** | 39360 |
| **Completed cells (terminal / latest)** | 39360 |
| **Failed cells (terminal / latest)** | 0 |
| **Skipped cells (manifest)** | 0 |
| **Duplicate historical rows** | 55 cell_ids with append-only failed→completed history (55 extra rows in `results.jsonl`) |
| **Integrity audit verdict** | SAFE TO FREEZE DATASET (`paper1_analysis/integrity_audit.md`) |
| **Python** | `3.12.13` (`CPython`) |
| **CALIPER version** | `0.1.0` |
| **Platform** | `Linux-6.8.0-136-generic-x86_64-with-glibc2.35` |

## Canonical analysis input

- **Primary:** `statistical_dataset.parquet` (39360 rows, 39360 unique cells, metric=`pass_at_1`)
- **Supporting evaluations:** `evaluations.parquet` (39360 unique cells)
- **Raw audit trail (do not analyze append-only failures as observations):** `results.jsonl` / `results.parquet`

## Policy

- Historical failed rows in `results.jsonl` **must remain untouched**.
- Confirmatory analyses **must** load `statistical_dataset.parquet` via `caliper.statistics.prepare.load_analysis_frame(..., require_statistical_dataset=True)`.
- This freeze pins the HumanEval+ full confirmatory measurement dataset for Paper 1.
