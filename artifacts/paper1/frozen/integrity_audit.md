# Integrity Audit — `paper1_confirmatory_humaneval_full`

**Auditor role:** Forensic recovery / freeze gate  
**Date:** 2026-08-03  
**Experiment directory:** `experiments/paper1_confirmatory_humaneval_full/paper1_confirmatory_humaneval_full`  
**Config:** `configs/paper1/confirmatory_humaneval_full.yaml`  
**Original run ID:** `c1b58081b70e`  
**Recovery run ID:** `b3fb840135b7`  
**Constraint compliance:** No inference re-run; no modification of completed observation payloads; repairs limited to derived artifacts and bookkeeping.

---

## Final verdict

# SAFE TO FREEZE DATASET

**Justification (evidence-backed):**

1. **Expected factorial size = 39 360** (6×2×4×164×5).  
2. **Latest-per-`cell_id` status = 39 360 completed, 0 failed.**  
3. **Checkpoints = 39 360 completed files, 0 failed.**  
4. **`inspect_missing_cells` → `missing_cell_ids = 0`.**  
5. **`statistical_dataset.parquet` = 39 360 unique completed cells** (includes recovered cell).  
6. **`evaluations.parquet` = 39 360 unique cells** (recovered cell merged).  
7. **`manifest.json` / `checkpoint_state.json` now agree** with reality (`completed_cells=39360`, `failed_cells=0`, `recovery_run_id=b3fb840135b7`).

The many `evaluate.skip_row` / `status_not_completed` log lines are **not** evidence of incomplete cells. They are emitted once per **historical failed row** still present in `results.jsonl` / `results.parquet` after successful retries (see §3).

---

## 1. Experiment state audit

| Metric | Value | Source |
|--------|------:|--------|
| Expected cells | **39 360** | config factorial axes |
| `results.jsonl` rows (all lines) | **39 415** | line parse |
| Unique `cell_id` | **39 360** | jsonl |
| Duplicated `cell_id` | **55** | jsonl (exactly one extra row each) |
| Extra duplicate rows | **55** | 39415 − 39360 |
| Status counts (all rows) | completed **39 360**, failed **55** | jsonl |
| Status counts (**latest per cell**) | completed **39 360**, failed **0** | jsonl last write |
| Other statuses | **none** | jsonl |
| Checkpoint files | **39 360** | `checkpoints/*.json` |
| Checkpoint completed / failed | **39 360 / 0** | `CheckpointStore` |
| `inspect_missing_cells` missing | **0** | post-repair |

### Recovered cell

| Field | Value |
|-------|-------|
| `cell_id` | `fa83232fabdb460d602e94f0bd8eff5aed5ed2434dab1c39f9b30ed9e96ab7c1` |
| Model | `qwen25_coder_7b` |
| Task | `task-humaneval_plus-070` |
| Prompt | `testing_oriented` |
| Temperature | `0.2` |
| Run index | `1` |
| First record | `failed` / `timed out` @ `2026-08-01T03:56:05Z` |
| Recovery record | `completed` / `score=1.0` @ `2026-08-03T16:04:09Z` |
| `recovery_run_id` | `b3fb840135b7` |
| Audit | `recovery_audit.jsonl` action=`recovered` |

Evidence: `recovery_audit.jsonl` contains **exactly one** entry; only this cell was executed by `retry-missing`.

---

## 2. Are `evaluate.skip_row` messages harmless or incomplete?

### Verdict: **A — harmless historical rows**

**Not B.** After recovery, **zero** cells remain incomplete under latest-per-cell semantics.

### Mechanism (code)

In `caliper/evaluation/runner.py`:

```python
for row in df.to_dict(orient="records"):
    if row.get("status") != "completed":
        logger.info("evaluate.skip_row", cell_id=..., reason="status_not_completed")
        continue
```

`evaluate_results_file` reads **all rows** from `results.parquet` (append-only history). It does **not** collapse to latest-per-`cell_id` before iterating.

### Why you saw ~55 skips after retrying one cell

| Fact | Count |
|------|------:|
| Historical `failed` rows still in `results.jsonl` | **55** |
| Of those, later overwritten by `completed` (same `cell_id`) | **55** |
| Still incomplete after latest write | **0** |

These 55 duplicates come from earlier timeouts that were later retried during the long main run / resumes — **not** from `retry-missing` expanding scope. `retry-missing` only executed the one missing cell in `missing_cells_report.json`.

When `finalize_experiment` re-runs evaluation over the full parquet, it logs `evaluate.skip_row` for **each** historical failed row → many log lines, even though only one cell was newly recovered.

---

## 3. Recovery pipeline audit

### `retry_missing.py` (what it does)

1. Load `missing_cells` from the report.  
2. Skip if already completed.  
3. `execute_cell_safe` for each missing cell; append to `results.jsonl`; write checkpoint.  
4. Append `recovery_audit.jsonl`.  
5. `writer.finalize()` → refresh `results.parquet`.  
6. `finalize_experiment(...)` → evaluate all rows, rebuild `statistical_dataset`, rewrite manifest/report/artifact.  
7. Update `checkpoint_state.json`; re-run `inspect_missing_cells`.

### What actually happened in this recovery

| Step | Evidence | Outcome |
|------|----------|---------|
| Recover cell | `recovery_audit.jsonl` action=`recovered`; checkpoint `status=completed`; results append | **OK** |
| Refresh `results.parquet` | mtime `2026-08-03 18:04`; 39 415 rows; target present as completed | **OK** |
| `finalize_experiment` evaluate + stats + manifest | Pre-repair: `statistical_dataset`/`evaluations`/`manifest` still at pre-recovery state (39 359 / failed=1 / no `recovery_run_id`) | **INCOMPLETE** |

**Root cause of inconsistency:** recovery **succeeded at the cell layer**, but **post-execution finalization did not complete** (manifest never gained `recovery_run_id`; derived tables omitted the recovered cell). Likely interruption/`SameFileError`-class failure path when finalizing with `config.yaml` already inside the experiment directory — the cell write had already committed.

This is **bookkeeping / derived-artifact drift**, not loss of the recovered observation.

---

## 4. Cross-file consistency (before → after repair)

### Before integrity repair (forensic snapshot)

| Artifact | State | Consistent with 39 360 completed? |
|----------|-------|-----------------------------------|
| `results.jsonl` | 39 415 rows; latest 39 360 completed | **Yes** (with historical fails) |
| `results.parquet` | same | **Yes** |
| Checkpoints | 39 360 completed | **Yes** |
| `statistical_dataset.parquet` | **39 359**; target **absent** | **No** |
| `evaluations.parquet` | **39 359** unique; target **absent** | **No** |
| `manifest.json` | completed=39359, failed=1; no recovery fields | **No** |
| `checkpoint_state.json` | completed=39359, failed=1 | **No** |

### After integrity repair

| Artifact | State |
|----------|-------|
| `statistical_dataset.parquet` | **39 360** unique; target present (`pass_at_1=1.0`) |
| `evaluations.parquet` / `.jsonl` | **39 360** unique; target metrics merged |
| `manifest.json` | completed=**39360**, failed=**0**, `recovery_run_id=b3fb840135b7` |
| `checkpoint_state.json` | completed=**39360**, failed=**0**, `recovery_run_id` set |
| `inspect_missing_cells` | missing=**0** |

**Unchanged (by design):** raw completed observations in `results.jsonl` / checkpoints; no re-inference.

**Backup of pre-repair derived files:**  
`_integrity_repair_backup_20260803T162625Z/`

---

## 5. Checkpoint audit

| Check | Result |
|-------|--------|
| Checkpoint file count | **39 360** |
| Missing checkpoints (vs expected IDs) | **0** |
| Failed checkpoints | **0** |
| Checkpoints without matching result `cell_id` | **0** |
| Result unique IDs without checkpoint | **0** |
| Orphan checkpoints | **0** |
| Orphan results (unique) | **0** |

Note: `results.jsonl` retains **55 historical failed rows** as append-only history. These are **not** orphans; each has a later completed row and a completed checkpoint.

---

## 6. Integrity summary table

| Item | Count |
|------|------:|
| Expected cells | 39 360 |
| Observed unique cells | 39 360 |
| Recovered cells (this recovery) | **1** |
| Remaining failed (latest) | **0** |
| Duplicated observations (failed→completed history) | **55** |
| Orphan checkpoints | **0** |
| Orphan unique results | **0** |
| Status distribution (latest) | completed 39360 |
| Status distribution (all rows) | completed 39360 + failed 55 |

---

## 7. Repair performed (smallest safe fix)

**Problem:** Derived artifacts + manifests lagged the successful cell recovery.

**Fix (implemented):**

1. Backup stale derived artifacts.  
2. Rebuild `statistical_dataset.parquet` from `results.parquet` rows with `status==completed` (39 360 cells; includes recovered cell; **does not alter scores of prior cells**).  
3. Evaluate **only** the recovered cell; **concat** its 5 metric rows into `evaluations.parquet` / `evaluations.jsonl` (no re-evaluation of the other 39 359 cells).  
4. Rewrite `manifest.json` and `checkpoint_state.json` to match latest-per-cell reality; record recovery metadata.  
5. Regenerate `report.md` and re-export artifact bundle.  
6. Re-run `inspect_missing_cells` → **0 missing**.

**Not done (intentionally):** regenerate experiment; rewrite completed predictions; full re-evaluation of all cells.

---

## 8. Why many `evaluate.skip_row` lines are expected

Whenever finalization evaluates `results.parquet` **without deduplicating** to latest-per-cell:

- Each of the **55** historical `failed` rows triggers one `evaluate.skip_row` (`reason=status_not_completed`).  
- This is independent of how many cells `retry-missing` executed (here: **1**).  
- Skipped rows are excluded from `evaluations*`; completed latest rows are what matter for analysis.

**Operational note for future recoveries:** expect ~N skip log lines if N historical failures exist in the append-only jsonl, even when `missing_cell_ids=0` after recovery.

---

## 9. Freeze recommendation

### SAFE TO FREEZE DATASET

Use as analysis inputs:

- **Primary:** `statistical_dataset.parquet` (39 360 completed `pass_at_1` rows)  
- **Supporting:** `evaluations.parquet`, `results.jsonl` / `results.parquet` (retain full audit trail including historical failures)  
- **Provenance:** `manifest.json`, `recovery_audit.jsonl`, `config.yaml`, checkpoints  

### Caveats to document in Paper 1 methods / threats

1. **Append-only duplicates:** 55 cells appear twice in raw results (failed then completed). Analysis must use **latest completed** or the normalized `statistical_dataset` (already deduplicated to one completed row per cell).  
2. **Recovery:** one timeout cell (`qwen25_coder_7b` × `task-humaneval_plus-070` × `testing_oriented` × `T=0.2` × `run_index=1`) was recovered under run `b3fb840135b7` and scored `pass_at_1=1.0`.  
3. **Artifact warning remaining:** `no experiment figures bundled in data/figures/` — unrelated to cell completeness.

### Do **not** freeze if you require

- Physically purged historical failed rows from `results.jsonl` (optional hygiene; **not required** for valid analysis), or  
- Bit-identical re-evaluation of all 39 360 cells in `evaluations.parquet` (current file is 39 359 original evals + 1 merged recovery eval).

---

## Appendix — key paths

```
experiments/paper1_confirmatory_humaneval_full/paper1_confirmatory_humaneval_full/
  results.jsonl
  results.parquet
  statistical_dataset.parquet          # freeze primary
  evaluations.parquet
  checkpoints/
  manifest.json
  checkpoint_state.json
  recovery_audit.jsonl
  missing_cells_report.json
  _integrity_repair_backup_20260803T162625Z/
  paper1_analysis/integrity_audit.md   # this file
```

---

*End of forensic integrity audit.*
