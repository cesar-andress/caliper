# Final repository audit snapshot (pre-change)

**Timestamp:** 2026-08-19T23:05+02:00  
**Absolute path:** `/home/cesar/papers/caliper/caliper`

## Git state

| Field | Value |
|-------|-------|
| Branch | `main` |
| HEAD | `66974ee813bc710ecd4bdfdc4a562f464e8f69b0` |
| Upstream | `origin/main` |
| Remote | `origin` → `git@github.com-ucjc:cesar-andress/caliper.git` |
| Stash | empty |
| Worktrees | 1 (default) |
| Submodules | none |

### Working tree (before audit commits)

Modified (18 files, +336/−40): v1.1.0 provider provenance / reasoning controls (uncommitted).

Untracked: `configs/paper1/paper1_confirmatory_humaneval_qwen3_v11_arm_{a,b}.yaml`, `configs/paper1/paper1_qwen3_v11_smoke.yaml`, `scripts/`, `tests/test_reasoning_controls.py`, `logs/`, `paper1_sibling_note.txt`.

## Tags

| Tag | Type | Target commit |
|-----|------|---------------|
| `v1.0.0` (local) | annotated | `66974ee813bc710ecd4bdfdc4a562f464e8f69b0` |
| `v1.0.0` (remote) | annotated | `66974ee813bc710ecd4bdfdc4a562f464e8f69b0` |

## Recent commits

```
66974ee docs: sync audit commit field to v1.0.0 HEAD
27b606e docs: set Zenodo audit commit to tagged HEAD
8366928 release: ship Paper 1 frozen statistical dataset for Zenodo/EMSE reuse
```

## Runtime / package

| Item | Value |
|------|-------|
| Python | 3.12.13 |
| Committed `__version__` | 1.0.0 |
| Working-tree `__version__` | 1.1.0 (uncommitted) |
| `pip show caliper` (editable install stale) | 1.0.0 |
| CLI `--version` (from source) | 1.1.0 |

## Release / Zenodo (verified)

| Item | State |
|------|-------|
| Zenodo DOI | `10.5281/zenodo.21780089` |
| Zenodo version label | v1.0.0 |
| Publication date | 2026-08-03 |
| Status | **Published** (immutable deposition) |
| GitHub release API | not accessible / not found via `gh` token |

## Critical file SHA256

| File | SHA256 |
|------|--------|
| `pyproject.toml` (working tree) | `744b45ebcb9392080176b5ce9ded35e5be5ef73714b49d78b57c1b4808a1686d` |
| `caliper/__init__.py` (working tree) | `60a41f6a85843e7d8a90920686b9c62a7bdc8f079f0725aad360f2614e1949b6` |
| `CITATION.cff` | `7d6d5cb47dada26553792e5660f7558ed5978a38bb9e74c002f77c5c26946641` |
| `artifacts/paper1/frozen/statistical_dataset.parquet` | `95209fff2f742d59b52aa5cf5616f1395ef7fc01fd87fe025c485577bca0d1c9` |

## Pre-audit test baseline (working tree, uncommitted v1.1)

- **303 passed**, 1 failed (`tests/test_cli.py::test_version` expects 1.0.0), 6 deselected integration/local

---

*Snapshot taken before audit commits. Subsequent commits document fixes in `FINAL_RELEASE_AUDIT.md`.*
