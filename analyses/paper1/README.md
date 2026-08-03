# Paper 1 analysis scripts

Scripts and Make targets for **Paper 1** analyses on factorial evaluation tables.

For the **frozen HumanEval+ confirmatory dataset** and independent reproduction
commands, use [`artifacts/paper1/`](../../artifacts/paper1/README.md) first.

## Scripts in this directory

| Script | Purpose |
|--------|---------|
| `variance_decomposition.py` | Descriptive variance components from a results table |
| `power_simulation.py` | Monte Carlo power simulation over design grids |
| `generate_publication_analysis.py` | Publication tables/figures from a completed experiment dir |
| `generate_robustness_analysis.py` | Robustness / GLMM exports |
| `generate_task_sampling_analysis.py` | Task-subset vs full-suite comparisons |
| `generate_design_guidance.py` | Design-guidance exports |

## Preferred reproduction path (frozen data)

```bash
python artifacts/paper1/scripts/verify_frozen_dataset.py
python artifacts/paper1/scripts/reproduce_paper1_core_tables.py
```

## CLI (generic tables)

```bash
caliper analyze variance --results path/to/evaluations.parquet
caliper analyze power --results path/to/evaluations.parquet
```

## Make targets (local experiment directories)

When a full local campaign directory exists under gitignored `experiments/`:

```bash
make paper1-humaneval-full-analysis
make paper1-humaneval-full-robustness
make paper1-humaneval-full-task-sampling
```

These targets expect `PAPER1_HUMANEVAL_FULL_DIR` (default:
`experiments/paper1_confirmatory_humaneval_full/paper1_confirmatory_humaneval_full`).

## Methods notes

- Sequential Type-I shares are descriptive and order-dependent; for the balanced
  Paper 1 freeze they are invariant under facet reordering (see manuscript).
- Confirmatory inference for binary pass@1 uses a binomial GLMM (logit); do not
  treat classical ANOVA-based $G$/$\Phi$ D-study exports as primary evidence.
- Crossed model-by-task interaction is reported via descriptive partitions in the
  manuscript reanalysis exports under
  `artifacts/paper1/analysis_exports/compliant_panel_reanalysis/`.

## Input schema

Tables should include `model`, `task_id`, `prompt_id`, `run_index` (or `run_id`),
`temperature`, `metric_name`, `metric_value`. The frozen statistical dataset
already conforms.
