# Current tag / release safety gate

**Date:** 2026-08-19

| Field | Value |
|-------|-------|
| Tag name | `v1.0.0` |
| Tag type | annotated |
| Old tag commit | `66974ee813bc710ecd4bdfdc4a562f464e8f69b0` |
| New HEAD (post-audit) | TBD after commit |
| Remote tag target (pre-push) | `66974ee813bc710ecd4bdfdc4a562f464e8f69b0` |

## GitHub release

- `gh release view v1.0.0`: not found / token lacks access
- Tag exists on remote

## Zenodo

- DOI: `10.5281/zenodo.21780089`
- Version label: **v1.0.0**
- Status: **Published** 2026-08-03 (immutable)
- Landing page archive: 740 KB source zip (pre–dataset-bundle note on page; repo `artifacts/paper1/` added in later commits on tag)

## Decision

**PUBLISHED_ZENODO_BLOCKS_TAG_REWRITE**

Rewriting `v1.0.0` to point at v1.1.0 commits would break provenance for the published Zenodo deposition cited in Paper 1.

## Safe options

1. **Recommended:** Keep `v1.0.0` @ `66974ee`; ship v1.1.0 on `main` without retagging Zenodo.
2. **Future (explicit author action):** Publish a new Zenodo version (e.g. v1.1.0) with new DOI/version record; update manuscript only if authors choose.
3. **Do not:** Force-move `v1.0.0` on published immutable Zenodo.

## Action taken

Tag **not** regenerated. Audit commits pushed to `main` only.
