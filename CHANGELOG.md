# Changelog

All notable changes to CALIPER are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] — 2026-08-19 (main branch; not re-tagged on Zenodo)

### Added

- Ollama reasoning controls: per-model `think`, `done_reason`, token counts, optional
  thinking digest/text, and `budget_exhausted` terminal status distinct from generic
  `completed` / `failed`.
- Raw provider response persistence under each run directory (`raw_responses/`).
- Post-freeze Paper 1 qwen3 diagnostic configs (`paper1_confirmatory_humaneval_qwen3_v11_arm_{a,b}.yaml`).
- Diagnostic scripts under `scripts/` for Arm A/B provenance and analysis (no LLM in CI).
- Tests: `tests/test_reasoning_controls.py`, expanded `tests/test_ollama_provider.py`.

### Changed

- Package version on `main` set to **1.1.0**; published Zenodo deposition remains **v1.0.0**
  (`10.5281/zenodo.21780089`) and must not be retroactively rewritten.

### Notes

- v1.0.0 behavior (Paper 1 confirmatory freeze) is historical; frozen parquet bytes unchanged.
- v1.1.0 capabilities apply to new runs only.

## [1.0.0] — 2026-08-03

### Added

- First public research release intended for Zenodo archival with Paper 1.
- Confirmatory HumanEval+ full-benchmark pipeline (164 tasks) with preparation,
  preflight, analysis, robustness/GLMM, task-sampling, and design-guidance Make targets.
- Missing-cell recovery and experiment path/status helpers for long factorial runs.
- Binomial mixed-model (GLMM) confirmatory analysis exports for `pass_at_1`.
- Task-sampling stability analysis and design-guidance exports.
- Release metadata: `LICENSE`, `CITATION.cff`, `.zenodo.json`, `codemeta.json`, `AUTHORS`,
  `docs/author_identity.md`, `RELEASE_NOTES_v1.0.0.md`.
- **`artifacts/paper1/`** freeze package: `statistical_dataset.parquet` (`N=39,360`),
  freeze/integrity/recovery docs, analysis CSV exports, protocol amendment comparison,
  and analysis-reproduction scripts (`verify_frozen_dataset.py`,
  `reproduce_paper1_core_tables.py`).
- Archival DOI: https://doi.org/10.5281/zenodo.21780089

### Changed

- Package version set consistently to **1.0.0**.
- Author metadata standardized to César Andrés, David Martín-Moncunill, and José Manuel Baños.
- README rewritten for EMSE-style artifact evaluation (Paper 1 reproduction path first-class).
- Release notes updated to state that the frozen statistical dataset **is** shipped.

### Notes

- Large raw campaign trees under `experiments/` remain local and gitignored.
- Public analysis reproduction uses `artifacts/paper1/`, not the full local experiment dump.
- Numeric classical G-theory D-study thresholds are not claimed; see Paper 1 manuscript
  and freeze audits.

## [0.1.0] — 2026-07

### Added

- Initial CALIPER artifact: factorial experiment runner, variance/power/ranking modules,
  Ollama local provider, mock pilots, and confirmatory benchmark scaffolding.
