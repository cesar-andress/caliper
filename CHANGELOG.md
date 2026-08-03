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
- Task-sampling stability analysis and evidence-supported design guidance exports.
- Release metadata: `LICENSE`, `CITATION.cff`, `.zenodo.json`, `codemeta.json`, `AUTHORS`,
  `docs/author_identity.md`, `RELEASE_NOTES_v1.0.0.md`.
- Archival DOI: https://doi.org/10.5281/zenodo.21780089

### Changed

- Package version set consistently to **1.0.0**.
- Author metadata standardized to César Andrés, David Martín-Moncunill, and José Manuel Baños.

### Notes

- Large experiment outputs under `experiments/` remain local and gitignored.
- Numeric G-theory D-study thresholds are not claimed by the analysis stack in this release;
  see Paper 1 analysis audits in the manuscript workspace.

## [0.1.0] — 2026-07

### Added

- Initial CALIPER artifact: factorial experiment runner, variance/power/ranking modules,
  Ollama local provider, mock pilots, and confirmatory benchmark scaffolding.
