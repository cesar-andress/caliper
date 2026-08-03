# Changelog

All notable changes to CALIPER are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
