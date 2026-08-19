# Paper1 qwen3 v1.1 scientific pipeline

## One command (after both arms complete)

```bash
cd caliper   # repository root
.venv/bin/python scripts/run_qwen3_v11_final_analysis.py
```

## Safe while arms are running

```bash
.venv/bin/python scripts/run_qwen3_v11_final_analysis.py --integrity-only
```

## Modules

| Module | Role |
|--------|------|
| `postrun_integrity.py` | Cell counts, metadata, raw responses, hashes |
| `scientific_validation.py` | A/B vs freeze and A vs B metrics + agreements |
| `heterogeneity.py` | Task difficulty, prompt/temp/run slices, bootstrap CIs |
| `statistical_comparison.py` | McNemar, paired bootstrap, Bland–Altman, Kendall/Spearman |
| `figures.py` | PNG/PDF figures |
| `provenance.py` | Software/git/Ollama/GPU/freeze hashes |
| `run_final_analysis.py` | Orchestrator with publication gate |

## Publication gate

Scientific outputs are written only when Arm A and Arm B both have 6560 cells
**and** `statistical_dataset.parquet`. Partial numbers are never authorized for
the manuscript unless `--allow-partial` is forced (forbidden for EMSE).
