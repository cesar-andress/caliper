# Final code audit report

**Date:** 2026-08-19  
**Repository:** `/home/cesar/papers/caliper/caliper`  
**Final commit:** `a9ccd85` (main)

## Summary

Full source audit of CALIPER with commit of previously uncommitted v1.1.0 provider-provenance work, documentation alignment with Paper 1, and test-suite validation.

## Bugs found

| Issue | Severity | Resolution |
|-------|----------|------------|
| v1.1.0 code uncommitted on `main` while tag stayed v1.0.0 | High (release drift) | Committed as `a9ccd85` |
| `test_cli.py` expected version 1.0.0 | Low | Updated to 1.1.0 |
| Hard-coded `/home/cesar/...` in script README | Low | Replaced with relative path |
| `logs/` not gitignored | Low | Added to `.gitignore` |

## Bugs fixed

- Provider provenance pipeline shipped on `main` (done_reason, eval_count, think, budget_exhausted, raw responses).
- Version/test consistency for 1.1.0 development line.

## Tests added

- `tests/test_reasoning_controls.py` (5 tests)
- Expanded `tests/test_ollama_provider.py`

## Final test results

```
304 passed, 6 deselected (integration/local), 1 warning (statsmodels FutureWarning)
Duration: ~19s
```

## Code-quality findings

- No production `TODO/FIXME/HACK` in `caliper/` package.
- No hard-coded `/home/cesar` in production code (one doc path fixed).
- Test API keys are mock placeholders only.

## Security findings

- `.env` gitignored; no real secrets in tracked files.
- **PASS**

## Reproducibility findings

- Frozen parquet SHA256 unchanged: `95209fff2f742d59b52aa5cf5616f1395ef7fc01fd87fe025c485577bca0d1c9`
- v1.0 tag commit `66974ee` preserved for Zenodo citation
- v1.1.0 documented as forward-looking on `main`

## Documentation changes

- `CHANGELOG.md` [1.1.0]
- `README.md` v1.0 Zenodo vs v1.1 main
- `docs/final_repository_architecture_audit.md`
- Audit snapshots and consistency docs

## Dependency changes

None (no upgrades).

## Commits

- `a9ccd85` feat: CALIPER v1.1.0 provider provenance and reasoning controls

## Push status

`main` pushed to `origin` (`66974ee..a9ccd85`).

## Tag policy

See `CURRENT_TAG_RELEASE_GATE.md`: **PUBLISHED_ZENODO_BLOCKS_TAG_REWRITE** — `v1.0.0` not moved.

## Unresolved (non-blocking)

- Zenodo landing page text still mentions 740 KB archive without dataset bundle (historical deposition).
- MBPP+ loader present; full 378-task EvalPlus differential suite not wired for Paper 1.
- Optional bib-year harmonization in manuscript repo (outside this audit).
