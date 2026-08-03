# CALIPER v1.0.0 — Release report

Date: 2026-08-03  
Repository: https://github.com/cesar-andress/caliper  
Branch pushed: `main`  
Tag pushed: `v1.0.0` (annotated)

## Commits created

| Hash | Message |
| --- | --- |
| `da539e2` | feat: add confirmatory HumanEval+ analysis and recovery pipeline |
| `860efbc` | docs: synchronize authorship and archival metadata for public release |
| `d333b4f` | release: prepare version 1.0.0 |

## Tag

| Item | Value |
| --- | --- |
| Tag name | `v1.0.0` |
| Annotated tag object | `9ba0eb02e62849aba75e0dccaabe1c3c5d568769` |
| Peeled commit | `d333b4f2e23be1be71d7a437559f6f3a2c20774a` |
| Annotation | CALIPER v1.0.0 — First public research release accompanying the empirical evaluation framework described in Paper 1. |
| Remote verification | `git ls-remote --tags origin` contains `refs/tags/v1.0.0` |

## Final version

**1.0.0** — consistent in `pyproject.toml`, `caliper/__init__.py`, CLI `--version`, `CITATION.cff`, `.zenodo.json`, README, CHANGELOG.

## Authors (canonical)

1. César Andrés — ORCID 0009-0001-8968-3404 — cesar.andress@ucjc.edu (corresponding)  
2. David Martín-Moncunill — ORCID 0000-0003-2422-9005 — david.martinm@ucjc.edu  
3. José Manuel Baños — ORCID 0009-0004-9971-7390  

## Sanity checks

- Unit tests: **298 passed**, 6 deselected (`integration`/`local`)
- Wheel build: `caliper-1.0.0-py3-none-any.whl`
- CLI: `caliper --version` → `1.0.0`; `caliper validate` on example config OK
- Experiment outputs not committed (`experiments/` gitignored)

## Remaining warnings

- Ruff reports style findings on some analysis modules (not blocking release)
- No GitHub Actions workflow yet (CI optional follow-up)
- Zenodo DOI not yet minted — update `CITATION.cff` / README after deposit
- Paper bibliographic DOIs intentionally omitted until acceptance

## Recommended GitHub Release notes

Title: `CALIPER v1.0.0`

Body: use `RELEASE_NOTES_v1.0.0.md`.

## Recommended Zenodo title

> CALIPER v1.0.0: Factorial LLM Evaluation Artifact for Variance-Aware Analysis (Paper 1)

Creators order and ORCIDs: see `.zenodo.json`.

## Archival readiness

**Ready for Zenodo archival** of the software artifact at tag `v1.0.0`, provided the operator uploads from this tag (or GitHub–Zenodo webhook) and mints the version DOI afterward.
