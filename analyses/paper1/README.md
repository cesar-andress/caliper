# Paper 1: Variance Decomposition and Statistical Power

This directory contains analysis scripts for **Paper 1** — variance decomposition
and statistical power in LLM evaluation.

## Scripts

| Script | Purpose |
|--------|---------|
| `variance_decomposition.py` | Descriptive stats, variance components, G/D studies |
| `power_simulation.py` | Monte Carlo power simulation over design grids |
| `generate_publication_analysis.py` | Publication tables, figures, LaTeX, and reports from a completed pilot |

## Publication analysis (Paper 1 pilot)

After the 6000-cell pilot completes:

```bash
make paper1-analysis
```

Outputs land in `experiments/paper1_ollama_pilot/paper1_analysis/` (`tables/`, `figures/`, `csv/`, `latex/`, `summary/`).

## Usage

```bash
# From repository root, after pip install -e .
python analyses/paper1/variance_decomposition.py \
  --results outputs/example_factorial/RUN_ID/evaluations.parquet

python analyses/paper1/power_simulation.py \
  --results outputs/example_factorial/RUN_ID/evaluations.parquet
```

Or via CLI:

```bash
caliper analyze variance --results evaluations.parquet
caliper analyze power --results evaluations.parquet
```

## Methods and Limitations

- **Sequential ANOVA** (`decompose_variance`): Type-I factor removal; order-dependent approximation.
- **MixedLM** (`fit_mixed_model`): Single grouping factor via statsmodels; falls back to ANOVA on failure.
- **G-theory**: Variance components from ANOVA; G and Φ coefficients for relative/absolute decisions.
- **Power simulation**: Monte Carlo two-model t-test under crossed task × prompt × run designs.

Crossed random effects (prompt × task × run) are not fully identified without specialized
mixed-model software. Treat ANOVA components as approximate; confirm with simulation.

## Input Schema

Results tables should include (aliases accepted):

- `model`, `task_id`, `prompt_id`, `run_id`, `temperature`
- `metric_name`, `metric_value`

Use `caliper evaluate` to produce evaluation tables from raw experiment results.
