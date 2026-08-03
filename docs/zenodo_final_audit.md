# Zenodo / artifact final audit — CALIPER v1.0.0

**Date:** 2026-08-03  
**Auditor role:** EMSE-style artifact evaluation  
**Repository:** https://github.com/cesar-andress/caliper  
**Concept / version DOI (published):** https://doi.org/10.5281/zenodo.21780089  

---

## 1. Files reviewed

| Path | Verdict |
|------|---------|
| `README.md` | Rewritten — Paper 1 reproduction first-class |
| `CITATION.cff` | Consistent authors/ORCIDs/DOI/v1.0.0 |
| `.zenodo.json` | Updated description to include freeze package |
| `codemeta.json` | Aligned description |
| `CHANGELOG.md` / `RELEASE_NOTES_v1.0.0.md` | Corrected “dataset not included” false claim |
| `AUTHORS` / `docs/author_identity.md` / `LICENSE` | Consistent |
| `pyproject.toml` / `caliper/__init__.py` | version `1.0.0` |
| `analyses/paper1/README.md` | Updated; points to `artifacts/paper1/` |
| `docs/RELEASE_AUDIT_v1.0.0.md` | Marked superseded (was stale 0.1.0 audit) |
| `docs/getting-started.md` | Paper 1 reproduction section added |
| `execute.sh` | Restored to runnable confirmatory-full wrapper |
| Zenodo landing page (pre-fix) | 740 KB source zip; **no** statistical dataset |

---

## 2. Critical reproducibility issue found

**Pre-fix Zenodo archive failed EMSE analysis reproduction:** it stated that experiment outputs were not bundled and contained only ~740 KB of source. Paper 1 analyses require `statistical_dataset.parquet` (`N=39,360`).

### Fix applied

Created tracked package:

`artifacts/paper1/` (~6 MB) containing:

- `frozen/statistical_dataset.parquet` — SHA-256 `95209fff2f742d59b52aa5cf5616f1395ef7fc01fd87fe025c485577bca0d1c9` (matches campaign freeze ledger)
- freeze / integrity / recovery / missing-cell docs
- locked config snapshot + version-controlled full YAML
- protocol comparison + documented 40→164 amendment note
- GLMM/variance CSV exports + compliant-panel reanalysis CSVs
- `scripts/verify_frozen_dataset.py` (executed OK)
- `scripts/reproduce_paper1_core_tables.py` (executed OK; compliant Type-I model 6.13%, task 42.37%)
- `SHA256SUMS` for packaged files

Raw multi-GB `experiments/` trees remain gitignored (not required for analysis reproduction).

---

## 3. Metadata issues found and fixed

| Issue | Fix |
|-------|-----|
| README / release notes claimed freeze not shipped | Corrected |
| `.zenodo.json` said “outputs not bundled” | Rewritten |
| `analyses/paper1/README` still pilot / classical G-theory primary | Rewritten |
| `execute.sh` empty / misleading | Restored |
| Stale pre-release audit readable as current | Superseded banner |
| Paper 1 amendment not packaged | Added `protocol/protocol_amendment_humaneval_full.md` |

Authors/ORCIDs/affiliation already matched `docs/author_identity.md` — reused, not invented.

Version strings for the package remain **1.0.0** / tag **v1.0.0**. Historical freeze docs correctly record that the campaign was frozen under software version `0.1.0` at freeze time; that is provenance, not a metadata bug.

---

## 4. Files modified / added (this audit)

- Added: `artifacts/paper1/**` (freeze package + scripts + SHA256SUMS)
- Modified: `README.md`, `CHANGELOG.md`, `RELEASE_NOTES_v1.0.0.md`, `.zenodo.json`, `CITATION.cff`, `codemeta.json`, `analyses/paper1/README.md`, `docs/getting-started.md`, `docs/RELEASE_AUDIT_v1.0.0.md`, `execute.sh`
- Added: `docs/zenodo_final_audit.md` (this file)

---

## 5. Git / release actions

(filled after commit/tag push)

| Field | Value |
|-------|-------|
| Commit | `77ac072934e6523d18c94eff45a9161ec818d97d` |
| Tag | `v1.0.0` (annotated; points at that commit) |
| Remote | `origin` (`github.com/cesar-andress/caliper`) |
| GitHub status | `main` + `v1.0.0` on origin. GitHub Release API returned 403 with the available token — recreate the GitHub Release from tag `v1.0.0` in the UI (or with a PAT that can write releases) so the Zenodo webhook re-archives the zip that includes `artifacts/paper1/`. |

---

## 6. Should Zenodo mint a new DOI/version?

**Yes — publish a refreshed GitHub Release for `v1.0.0` (or a new tag if the Zenodo–GitHub integration refuses tag reuse).**

Reason: the archival zip must include `artifacts/paper1/`. The previously published Zenodo record (740 KB, no freeze) is scientifically incomplete for Paper 1 analysis reproduction.

- If GitHub→Zenodo creates a **new version**, update the manuscript DOI badge only if the version DOI changes; prefer the **concept DOI** for long-term citation if Zenodo provides one distinct from `10.5281/zenodo.21780089`.
- Re-check the Zenodo landing page after the webhook runs and confirm `artifacts/paper1/frozen/statistical_dataset.parquet` is present in the downloaded archive.

---

## 7. Remaining limitations (honest)

1. Full raw completion dump (`results.jsonl`) is not shipped — analysis uses `statistical_dataset.parquet` (includes `prediction` text for compliance audits).
2. Exact quantization digests were not recorded; Ollama tag replay may drift.
3. End-to-end re-execution of 39,360 local inferences is not a one-command CI job (hardware/models required).
4. `docs/issues.md` / `docs/roadmap.md` remain internal backlog documents with historical “placeholder” language — not release blockers.

---

## 8. Would this artifact satisfy an EMSE artifact evaluation?

**After this fix, for analysis reproduction of Paper 1: yes (functional / reusable), contingent on the refreshed Zenodo zip actually containing `artifacts/paper1/`.**

Before this fix: **no** — the published archive could not reproduce Paper 1 tables.

What an AE can now do from the archive alone:

1. Install the package  
2. Verify the freeze checksum and `N=39,360`  
3. Regenerate core Type-I / compliance tables  
4. Inspect locked GLMM exports and protocol amendment docs  

What still requires author-scale resources (acceptable if documented): bitwise re-execution of the Ollama campaign.
