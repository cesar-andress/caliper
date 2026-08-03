# Release audit — CALIPER (pre-v1.0.0)

> **Superseded.** This document records the *pre-release* audit state (version still
> 0.1.0, missing LICENSE/CITATION, no freeze package). For the post-fix archival
> audit, see [`docs/zenodo_final_audit.md`](zenodo_final_audit.md).

Date: 2026-08-03  
Repository: `~/papers/caliper/caliper`  
Remote: `git@github.com-ucjc:cesar-andress/caliper.git`  
Branch: `main` @ `4792dc6` (tracking `origin/main`)

## Git snapshot

| Item | Finding |
| --- | --- |
| Default branch | `main` |
| Ahead/behind | Up to date with `origin/main` before release commits |
| Existing tags | None |
| Working tree | Substantial uncommitted confirmatory-pipeline work + untracked analysis modules |
| Experiment outputs | Present locally under `experiments/` (~1.3 G) — **gitignored** (must not be committed) |
| `.venv/` | Present locally — **gitignored** |

## Release metadata inventory

| Artifact | Status |
| --- | --- |
| README.md | Present; version **0.1.0**; placeholder authors / GitHub URL |
| LICENSE | **Missing** (MIT declared in `pyproject.toml`) |
| CITATION.cff | **Missing** |
| CHANGELOG / RELEASE notes | **Missing** |
| AUTHORS / CONTRIBUTORS | **Missing** |
| NOTICE / COPYING | Absent (OK if MIT LICENSE present) |
| `.zenodo.json` | **Missing** |
| `codemeta.json` | Absent (optional) |
| `pyproject.toml` | Present; version **0.1.0**; authors = `{CALIPER Team}` |
| `setup.cfg` / `setup.py` | Absent (hatchling via pyproject — OK) |
| `caliper/__init__.py` | `__version__ = "0.1.0"` |
| GitHub Actions | **Missing** (no `.github/`) |
| Documentation | `docs/` present; getting-started/architecture OK; roadmap/issues are internal backlog |
| Examples | `configs/examples/`, `data/examples/` present |
| Makefile | Present; Paper 1 confirmatory targets present |

## Author identity (canonical — to apply)

From project author-identity standardization and sibling UCJC research artifacts (`vsdlc`, `localgovbench`):

| Order | Display name | ORCID | Email |
| ---: | --- | --- | --- |
| 1 | César Andrés (corresponding) | 0009-0001-8968-3404 | cesar.andress@ucjc.edu |
| 2 | David Martín-Moncunill | 0000-0003-2422-9005 | david.martinm@ucjc.edu |
| 3 | José Manuel Baños | 0009-0004-9971-7390 | (affiliation UCJC; email not required in all sibling deposits) |

Shared affiliation: CRIA-BDHS Research Group, Higher Polytechnic School of Technology and Science, Universidad Camilo José Cela, Madrid, Spain.

**Current defect:** repository metadata uses anonymous/`CALIPER Team` / `Author, A.` placeholders.

## Hygiene findings

| Check | Result |
| --- | --- |
| Merge conflict markers | None found |
| Broken symlinks | None found |
| Secrets in tracked tree | No API keys observed in quick scan |
| Temporary/cache | `.pytest_cache/` present locally (gitignored) |
| `execute.sh` | Tracked but **empty (0 bytes)** — remove or restore |
| Version consistency | All package version strings still **0.1.0** |
| Mixed versions | None beyond 0.1.0 placeholders |

## Uncommitted change classes (for atomic commits)

1. **feat/build:** confirmatory HumanEval+ pipeline, missing-cell recovery, GLMM/task-sampling/design-guidance analysis modules, CLI extensions, tests, configs
2. **docs:** README/citation/authors/release notes/Zenodo metadata/LICENSE/CHANGELOG
3. **release:** bump to **v1.0.0** across `pyproject.toml`, `__init__.py`, CITATION/Zenodo

## Risks / non-goals

- Do **not** commit `experiments/` statistical datasets or model outputs
- Do **not** rewrite Related Work / Paper 1 LaTeX (out of this repo’s public artifact scope except citation pointers)
- Do **not** force-push or delete tags

## Verdict of Phase 1

Repository is **not yet archival-ready**: missing LICENSE/CITATION/Zenodo/AUTHORS, placeholder authorship, version still 0.1.0, and uncommitted scientific pipeline code must be included in the public release commits.
