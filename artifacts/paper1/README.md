# Paper 1 frozen artifact package

This directory is the **analysis-reproduction bundle** for Paper 1
(*Pass@1 Is a Measurement: Facet Accounting for HumanEval+-Style Function-Level
Code Synthesis Evaluation*).

It is intentionally small: it includes the frozen statistical analysis dataset
and provenance documents required to reproduce the paper’s tables **without**
re-running local LLM inference.

## Contents

| Path | Role |
|------|------|
| `frozen/statistical_dataset.parquet` | Canonical analysis input (`N=39,360`, metric `pass_at_1`) |
| `frozen/config.yaml` | Locked experiment configuration snapshot |
| `frozen/manifest.json` | Run manifest (completed cells, recovery run id) |
| `frozen/recovery_audit.jsonl` | Recovery audit trail |
| `frozen/paper1_dataset_freeze*.md` / `*_checksums.txt` | Freeze documentation and original checksum ledger |
| `frozen/integrity_audit.md` | Freeze-gate integrity audit (`SAFE TO FREEZE`) |
| `frozen/analysis_manifest.md` | Analysis export inventory |
| `analysis_exports/` | Locked GLMM/variance CSV exports + compliant-panel reanalysis CSVs |
| `protocol/confirmatory_humaneval_full.yaml` | Version-controlled full HumanEval+ config |
| `protocol/paper1_humaneval_full_protocol_comparison.*` | 40-task vs 164-task protocol comparison (documented amendment) |
| `scripts/verify_frozen_dataset.py` | Checksum + row-count verification |
| `scripts/reproduce_paper1_core_tables.py` | Regenerates core Type-I / compliance CSVs from the freeze |
| `SHA256SUMS` | Digests for all packaged files |

## What is not included

- Full append-only `results.jsonl` / raw completion dumps under `experiments/`
  (large; local-only; not required for analysis reproduction from
  `statistical_dataset.parquet`)
- Model weights and Ollama blobs
- Manuscript LaTeX sources (companion workspace)

Execution reproduction of the factorial (re-running 39,360 local inferences)
requires matching served Ollama tags, hardware, and the configs under
`configs/paper1/`. Exact on-disk quantization digests were not recorded in the
freeze metadata.

## Verify and reproduce (analysis)

From the CALIPER repository root, after `pip install -e .`:

```bash
python artifacts/paper1/scripts/verify_frozen_dataset.py
python artifacts/paper1/scripts/reproduce_paper1_core_tables.py
```

Expected verify output includes:

- `sha256 = 95209fff2f742d59b52aa5cf5616f1395ef7fc01fd87fe025c485577bca0d1c9`
- `rows = 39360`

Optional full checksum check of the package:

```bash
cd artifacts/paper1 && sha256sum -c SHA256SUMS
```

## Relation to gitignored `experiments/`

Developers who executed the campaign locally still have the full experiment
directory under `experiments/paper1_confirmatory_humaneval_full/` (gitignored).
The files in `artifacts/paper1/frozen/` are the subset that must ship with the
public archive so that independent readers can audit Paper 1 analyses.
