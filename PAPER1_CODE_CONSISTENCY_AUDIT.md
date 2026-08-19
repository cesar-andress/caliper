# Paper 1 ↔ CALIPER code consistency audit

**Date:** 2026-08-19  
**Manuscript:** sibling `../paper1/` (final EMSE pass)  
**Software:** this repository

| Item | Manuscript claim | Repository state | Verdict |
|------|------------------|------------------|---------|
| HumanEval+ tasks | 164 | `humaneval_plus.py` + frozen parquet | **MATCH** |
| Prompts | 4 | confirmatory YAML | **MATCH** |
| Temperatures | 0.0, 0.2 | confirmatory YAML | **MATCH** |
| Runs | 5 | confirmatory YAML | **MATCH** |
| Full cells | 39,360 | frozen parquet N | **MATCH** |
| Compliant panel | 32,800 (5 models) | analysis scripts | **MATCH** |
| Frozen dataset SHA256 | 95209fff… | `artifacts/paper1/frozen/` | **MATCH** |
| v1.0 freeze metadata | no done_reason/eval_count | v1.0 tag behavior | **DOCUMENTED_HISTORICAL_DIFFERENCE** |
| qwen3 think default | Ollama ON, not sent in YAML | forensic + v1.0 code at tag | **DOCUMENTED_HISTORICAL_DIFFERENCE** |
| Post-freeze diagnostics | CALIPER 1.1.0 | `main` v1.1.0 uncommitted→committed | **MATCH** (after commit) |
| Zenodo DOI | 10.5281/zenodo.21780089 | CITATION.cff, README | **MATCH** |
| Release tag | v1.0.0 | annotated tag @66974ee | **MATCH** |
| Arms A/B location | repo outputs paths | `scripts/` + configs | **MATCH** |
| MBPP+ replication | not executed for Paper 1 | loader only | **DOCUMENTED_HISTORICAL_DIFFERENCE** |

**Unexplained MISMATCH entries:** 0
