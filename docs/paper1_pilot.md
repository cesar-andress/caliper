# Paper 1 mock pilot — variance decomposition

This pilot validates the **full CALIPER analysis pipeline** for Paper 1
(variance decomposition and statistical power) **before** running real API models.

It is **not** final scientific evidence. Outputs demonstrate that the
factorial runner, evaluation layer, statistical tables, power simulation,
ranking fragility analysis, and artifact export work end-to-end on a
realistic design size.

## Objective

- Exercise a **6000-cell** factorial experiment under controlled mock model tiers.
- Produce analysis-ready tables for variance decomposition and power simulation.
- Verify reproducibility artifacts (`manifest.json`, `report.md`, `artifact/`).

## Factorial design

| Factor | Levels |
|--------|--------|
| Models | `mock_strong`, `mock_medium`, `mock_noisy` |
| Tasks | 20 Python code-generation items (`data/paper1/pilot_code_tasks.jsonl`) |
| Prompts | `direct`, `concise`, `step_by_step`, `test_aware`, `production_quality` |
| Temperatures | `0.0`, `0.7` |
| Runs | 10 |

**Total cells:** 3 × 20 × 5 × 2 × 10 = **6000**

### Simulated model quality

| Model | Provider | `match_rate` | Behavior |
|-------|----------|--------------|----------|
| `mock_strong` | mock | 0.85 | High exact-match rate, deterministic |
| `mock_medium` | mock | 0.55 | Moderate exact-match rate |
| `mock_noisy` | random | 0.25 | Stochastic low-quality responses |

`match_rate` controls how often the provider returns the reference solution
(when available in request metadata). This injects realistic score variance
across models, prompts, tasks, and runs without API calls.

## How to run

From the software artifact directory (`caliper/`):

```bash
# Validate design
caliper validate --config configs/paper1/pilot_variance_decomposition.yaml
caliper plan     --config configs/paper1/pilot_variance_decomposition.yaml

# Execute (resumable; checkpoints enabled)
caliper run --config configs/paper1/pilot_variance_decomposition.yaml
```

Output directory (default layout):

```
experiments/paper1_pilot_variance/
├── config.yaml
├── results.parquet
├── statistical_dataset.parquet
├── evaluations.parquet
├── manifest.json
├── report.md
└── artifact/
```

## How to analyze

After the run completes:

```bash
EXP=experiments/paper1_pilot_variance

# Paper 1 — variance decomposition
caliper analyze variance --results "$EXP/statistical_dataset.parquet"

# Paper 1 — power simulation
caliper analyze power --results "$EXP/statistical_dataset.parquet"

# Paper 2 — ranking fragility (pipeline sanity check)
caliper ranking-fragility "$EXP/results.parquet" \
  --output-dir "$EXP/analysis/ranking_fragility" \
  --reports-dir "$EXP/figures/ranking_fragility"

# Re-export or verify artifact bundle
caliper export-artifact "$EXP"
```

Or use the bundled script:

```bash
cd experiments/paper1_pilot_variance/artifact
./reproduce.sh
```

## Expected output files

| File | Purpose |
|------|---------|
| `results.parquet` | Raw factorial cell records |
| `statistical_dataset.parquet` | Paper 1 normalized schema |
| `evaluations.parquet` | Post-hoc metric evaluation |
| `manifest.json` | Provenance and environment metadata |
| `report.md` | Human-readable execution summary |
| `artifact/` | OSF-style reproduction bundle |
| `analysis/ranking_fragility/` | Bootstrap ranking outputs (after CLI step) |

## Limitations

1. **Mock providers only** — no real LLM inference; scores reflect configured
   `match_rate`, not model capability.
2. **Heuristic test scoring** — `test_pass_rate` is a string-matching proxy;
   code is not executed.
3. **No confirmatory claims** — use this pilot to validate infrastructure,
   not to support paper conclusions.
4. **6000 cells** — larger than unit-test configs but still small vs. planned
   confirmatory studies.
5. **Single domain** — all tasks are short Python code generation from a
   synthetic CALIPER pilot benchmark.

## Next steps (real Paper 1 study)

1. Replace mock tiers with API/local model configs.
2. Expand tasks to preregistered public benchmarks.
3. Run confirmatory design with pre-registered seeds and analysis scripts.
4. Archive frozen artifact for OSF submission.
