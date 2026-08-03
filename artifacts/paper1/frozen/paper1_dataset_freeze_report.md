# Paper 1 Dataset Freeze Report

**Verdict:** PAPER 1 DATASET FROZEN  
**Freeze timestamp (UTC):** `2026-08-03T16:34:29.887883+00:00`  
**Experiment:** `paper1_confirmatory_humaneval_full`  
**Path:** `experiments/paper1_confirmatory_humaneval_full/paper1_confirmatory_humaneval_full`

---

## Dataset summary

| Item | Value |
|------|------:|
| Expected cells | 39360 |
| Terminal completed (latest-per-cell) | 39360 |
| Terminal failed | 0 |
| Manifest skipped | 0 |
| `results.jsonl` rows | 39415 |
| Historical duplicate cell_ids (failed→completed) | 55 |
| `statistical_dataset.parquet` rows | 39360 |
| `evaluations.parquet` unique cells | 39360 |
| Run ID | `c1b58081b70e` |
| Recovery run ID(s) | `b3fb840135b7` |

Overall pass@1 (frozen statistical dataset): **0.657088**

## Integrity summary

- Pre-freeze gate: `VERIFY_OK` (manifest ↔ checkpoints ↔ jsonl latest ↔ stats ↔ evals).
- Prior forensic audit: `paper1_analysis/integrity_audit.md` → **SAFE TO FREEZE DATASET**.
- Historical failed rows retained in append-only `results.jsonl` / `results.parquet` (**untouched**).

## Reproducibility summary

### Analysis input policy (enforced in code)

- Added `caliper.statistics.prepare.load_analysis_frame(..., require_statistical_dataset=True)`.
- Publication / design-guidance / robustness / task-sampling loaders now **require** `statistical_dataset.parquet`.
- CLI helpers `variance_decomposition.py` / `power_simulation.py` apply `completed_rows_only()` before analysis if raw dumps are passed.

### Regenerated under freeze

- `paper1_analysis/design_guidance/*` regenerated from frozen `statistical_dataset.parquet` (replacing pre-completion placeholders).

### Not regenerated in this freeze step (optional post-freeze)

- Full `make paper1-humaneval-full-analysis` publication tables/figures
- `make paper1-humaneval-full-robustness`
- `make paper1-humaneval-full-task-sampling`

These **must** be generated from the frozen `statistical_dataset.parquet` and do **not** require re-inference.

## Freeze summary

| Artifact | Role |
|----------|------|
| `paper1_dataset_freeze.md` | Freeze metadata |
| `paper1_dataset_freeze_checksums.txt` | SHA256 inventory |
| `paper1_dataset_freeze_report.md` | This report |
| `paper1_analysis/analysis_manifest.md` | Analysis file inventory |
| `paper1_analysis/integrity_audit.md` | Forensic gate |

**Software:** Python `3.12.13`, CALIPER `0.1.0`, git `main@4792dc6004e8`

## Remaining known limitations

1. Append-only history retains **55** failed→completed duplicates in raw results; analyses must use `statistical_dataset.parquet`.
2. Full publication/robustness/task-sampling figure bundles are **not** all present yet under `paper1_analysis/` (design guidance + integrity audit + analysis manifest are).
3. Artifact export may still warn about missing `figures/` until publication analysis is run.
4. One cell was recovery-completed (`recovery_run_id=b3fb840135b7`); documented in freeze metadata.
5. D-study G coefficients in design guidance may be numerically flat under the current estimator (known limitation; does not affect freeze of raw/pass@1 observations).

## Canonical statement

**This frozen HumanEval+ confirmatory dataset (`paper1_confirmatory_humaneval_full`, run `c1b58081b70e`, 39360 completed cells in `statistical_dataset.parquet`) is the canonical measurement dataset for CALIPER Paper 1 full-benchmark confirmatory analyses.**

The 40-task subset study (`paper1_confirmatory_humaneval`) remains a separate frozen confirmatory tier and is not superseded for subset-specific reporting.

---

## Git state (no commit performed)

```
M Makefile
 M analyses/paper1/generate_publication_analysis.py
 M analyses/paper1/power_simulation.py
 M analyses/paper1/variance_decomposition.py
 M caliper/benchmarks/experiment_yaml.py
 M caliper/benchmarks/materialize.py
 M caliper/cli.py
 M caliper/runners/checkpoint.py
 M caliper/runners/experiment.py
 M caliper/statistics/glmm_analysis.py
 M caliper/statistics/prepare.py
 M caliper/statistics/robust_analysis.py
 M caliper/statistics/robustness_report.py
 M caliper/validation/config_builder.py
 M caliper/validation/confirmatory.py
 M execute.sh
?? analyses/paper1/generate_design_guidance.py
?? analyses/paper1/generate_task_sampling_analysis.py
?? caliper/runners/experiment_paths.py
?? caliper/runners/experiment_status.py
?? caliper/runners/failures.py
?? caliper/runners/missing_cells.py
?? caliper/runners/retry_missing.py
?? caliper/statistics/design_guidance.py
?? caliper/statistics/task_sampling.py
?? caliper/validation/protocol_comparison.py
?? configs/paper1/confirmatory_humaneval_full.yaml
?? docs/paper1_humaneval_full_protocol_comparison.json
?? docs/paper1_humaneval_full_protocol_comparison.md
?? tests/test_experiment_paths.py
?? tests/test_glmm_analysis.py
?? tests/test_humaneval_full_extension.py
?? tests/test_missing_cells.py
```

### Unexpected / pre-existing dirty paths (not created solely by this freeze)

The working tree contains many modified/untracked files beyond freeze metadata (runners, validation, tests, configs, etc.). Review carefully before committing. Freeze-specific additions are under the experiment directory (`paper1_dataset_freeze*`, updated `paper1_analysis/`) plus analysis-loader hardening in `caliper/statistics/prepare.py` and related callers.

### Suggested commit commands (review first; do **not** run blindly)

```bash
cd ~/papers/caliper/caliper
git status

# Minimal freeze commit (metadata + hardened loaders + analysis inventory)
git add \
  experiments/paper1_confirmatory_humaneval_full/paper1_confirmatory_humaneval_full/paper1_dataset_freeze.md \
  experiments/paper1_confirmatory_humaneval_full/paper1_confirmatory_humaneval_full/paper1_dataset_freeze_checksums.txt \
  experiments/paper1_confirmatory_humaneval_full/paper1_confirmatory_humaneval_full/paper1_dataset_freeze_report.md \
  experiments/paper1_confirmatory_humaneval_full/paper1_confirmatory_humaneval_full/paper1_analysis/ \
  experiments/paper1_confirmatory_humaneval_full/paper1_confirmatory_humaneval_full/manifest.json \
  experiments/paper1_confirmatory_humaneval_full/paper1_confirmatory_humaneval_full/checkpoint_state.json \
  experiments/paper1_confirmatory_humaneval_full/paper1_confirmatory_humaneval_full/statistical_dataset.parquet \
  experiments/paper1_confirmatory_humaneval_full/paper1_confirmatory_humaneval_full/evaluations.parquet \
  experiments/paper1_confirmatory_humaneval_full/paper1_confirmatory_humaneval_full/evaluations.jsonl \
  experiments/paper1_confirmatory_humaneval_full/paper1_confirmatory_humaneval_full/recovery_audit.jsonl \
  experiments/paper1_confirmatory_humaneval_full/paper1_confirmatory_humaneval_full/report.md \
  caliper/statistics/prepare.py \
  caliper/statistics/design_guidance.py \
  caliper/statistics/task_sampling.py \
  caliper/statistics/robustness_report.py \
  analyses/paper1/generate_publication_analysis.py \
  analyses/paper1/variance_decomposition.py \
  analyses/paper1/power_simulation.py

# Optional large binaries (only if remote policy allows LFS/large files):
# git add experiments/paper1_confirmatory_humaneval_full/paper1_confirmatory_humaneval_full/results.jsonl
# git add experiments/paper1_confirmatory_humaneval_full/paper1_confirmatory_humaneval_full/results.parquet
# git add experiments/paper1_confirmatory_humaneval_full/paper1_confirmatory_humaneval_full/checkpoints/

git commit -m "$(cat <<'EOF'
Freeze Paper 1 HumanEval+ full confirmatory dataset and analysis loaders.

Pin statistical_dataset (39360 cells), freeze checksums/metadata, require
statistical_dataset for confirmatory analyses, and document recovery provenance.
EOF
)"
```
