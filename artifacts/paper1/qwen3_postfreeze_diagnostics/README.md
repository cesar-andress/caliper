# POST-FREEZE DIAGNOSTIC EVIDENCE — Qwen3 / HumanEval+ (Paper 1)

**Role:** companion archive for RQ2 instrumentation claims.  
**NOT part of the confirmatory HumanEval+ freeze.**

## Confirmatory freeze (immutable)

- Software / configs DOI: https://doi.org/10.5281/zenodo.21780089 (v1.0.0)
- Confirmatory analysis dataset SHA-256 (`statistical_dataset.parquet`):
  `95209fff2f742d59b52aa5cf5616f1395ef7fc01fd87fe025c485577bca0d1c9`
- This package does **not** rewrite that freeze.

## What this package contains

1. **arm_a/** — think-off, 1024-token diagnostic (N=6560): provider fields
   (`done_reason`, `eval_count`, think/budget flags), empty flag, latency, pass@1.
   Full completion/thinking text omitted; lengths/hashes retained.
2. **arm_b/** — think-on, 4096-token sequential diagnostic (final stored n=801)
   with the same provider fields and sequential summary.
3. **freeze_forensics/** — derived freeze-era Qwen3 cell extract (N=6560):
   empty flag, locked vs relaxed extractability, character length, client latency.
   Matches Appendix A (empty ≈76.4%, relaxed recovers 0 cells, max nonempty length 642).
4. **protocols/** — Arm A/B YAML configs.
5. **audits/** — supporting out-of-freeze diagnostic JSON used in the forensic narrative.

## Why these diagnostics were run

The confirmatory freeze scored mass empty visible completions for Qwen3 32B under
the locked contract but did not persist provider termination metadata. Arms A/B
probe configuration sensitivity after the freeze; Appendix A forensics test whether
empties are recoverable by extraction alone.

## Not primary inference

Arms A/B and the forensic extract are **outside confirmatory N**. They do not
restore a six-model confirmatory panel and do not identify a think-flag causal
effect at fixed budget under a fixed harness.

## Reproduce reported quantities

- Arm A empty rate / pass@1 / done_reason: `arm_a/arm_a_cell_table.parquet`
- Arm B θ̂ = 145/800 and statuses: `arm_b/arm_b_cell_table.parquet` + `arm_b_summary.json`
- Appendix A: `freeze_forensics/qwen3_freeze_forensic_summary.json`

## Provenance limitations

- Freeze-era thinking state was not logged (inferred from Ollama defaults).
- Model-weight quantization digests were not recorded.
- Raw completion dumps / `results.jsonl` are not republished here; the forensic
  extract is the minimum auditable derivative for Appendix A.

## Software metadata

- Diagnostics generated with CALIPER 1.1.0 (post-freeze).
- Confirmatory freeze generated with the freeze-era CALIPER build archived under
  Zenodo v1.0.0.
