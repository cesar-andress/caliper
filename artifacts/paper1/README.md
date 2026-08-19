# Paper 1 frozen artifact package

This directory is the **analysis-reproduction bundle** for Paper 1
(*Model–Task Heterogeneity and Instrumentation Distortion in HumanEval+ Pass@1
Evaluation*).

It includes the frozen confirmatory statistical analysis dataset and provenance
documents required to reproduce the paper’s confirmatory tables **without**
re-running local LLM inference, plus a clearly separated post-freeze diagnostic
package for RQ2.

## Contents

| Path | Role |
|------|------|
| `frozen/statistical_dataset.parquet` | **CONFIRMATORY** analysis input (`N=39,360`) |
| `frozen/config.yaml` | Locked experiment configuration snapshot |
| `frozen/manifest.json` | Run manifest (completed cells, recovery run id) |
| `frozen/recovery_audit.jsonl` | Recovery audit trail |
| `frozen/paper1_dataset_freeze*.md` / `*_checksums.txt` | Freeze documentation and file-integrity checksum ledger |
| `frozen/integrity_audit.md` | Freeze-gate integrity audit |
| `frozen/analysis_manifest.md` | Analysis export inventory |
| `analysis_exports/` | Locked GLMM/variance CSV exports + compliant-panel reanalysis CSVs |
| `qwen3_postfreeze_diagnostics/` | **POST-FREEZE DIAGNOSTIC EVIDENCE** (Arms A/B + Appendix A extract) |
| `protocol/` | Version-controlled full HumanEval+ config and amendment notes |
| `scripts/verify_frozen_dataset.py` | Checksum + row-count verification |
| `scripts/reproduce_paper1_core_tables.py` | Regenerates core Type-I / compliance CSVs from the freeze |
| `SHA256SUMS` | Digests for packaged confirmatory files |

## Confirmatory vs post-freeze

- **Confirmatory freeze** (`frozen/`): immutable primary evidence for RQ1 and the
  descriptive partitions. SHA-256 of `statistical_dataset.parquet`:
  `95209fff2f742d59b52aa5cf5616f1395ef7fc01fd87fe025c485577bca0d1c9`.
- **Post-freeze diagnostics** (`qwen3_postfreeze_diagnostics/`): Arm A/B and the
  Qwen3 forensic cell extract for Appendix A / RQ2. **Not** merged into
  confirmatory $N$. See that directory’s `README.md`.

## What is not included

- Full append-only `results.jsonl` / unrestricted raw completion dumps under
  `experiments/` (not required to audit confirmatory tables or Appendix A once
  the forensic extract is present)
- Model weights and Ollama blobs
- Model-weight / quantization digests (never recorded)
- Manuscript LaTeX sources (companion workspace)

## Verify and reproduce (analysis)

From the CALIPER repository root, after `pip install -e .`:

```bash
python artifacts/paper1/scripts/verify_frozen_dataset.py
python artifacts/paper1/scripts/reproduce_paper1_core_tables.py
```

Expected verify output includes:

- `sha256 = 95209fff2f742d59b52aa5cf5616f1395ef7fc01fd87fe025c485577bca0d1c9`
- `rows = 39360`

Optional full checksum check of the confirmatory package files:

```bash
cd artifacts/paper1 && sha256sum -c SHA256SUMS
```

## Relation to gitignored `experiments/`

Developers who executed the campaign locally still have the full experiment
directory under `experiments/paper1_confirmatory_humaneval_full/` (gitignored).
The files in `artifacts/paper1/frozen/` are the subset that must ship with the
public archive so that independent readers can audit Paper 1 analyses.
