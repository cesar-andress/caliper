# Final repository architecture audit

**Repository:** CALIPER (`caliper/`)  
**Date:** 2026-08-19

## Top-level layout

| Path | Role |
|------|------|
| `caliper/` | Python package (CLI, runners, models, evaluation, statistics, benchmarks) |
| `configs/` | Experiment YAML (examples + Paper 1 confirmatory + qwen3 v1.1 arms) |
| `tests/` | Pytest suite (303+ tests; integration/local deselected by default) |
| `artifacts/paper1/` | **AUTHORITATIVE** frozen Paper 1 dataset + reproduction scripts |
| `analyses/paper1/` | Analysis documentation pointers |
| `docs/` | User guides, release audits, author identity |
| `scripts/` | **DIAGNOSTIC** Paper 1 qwen3 v1.1 post-freeze tooling |
| `Makefile` | Confirmatory prep/run/analysis targets |

## Execution pipeline

```
YAML config → config/schema.py (validation)
           → runners/experiment.py (plan cells)
           → runners/executor.py (provider + task scoring)
           → runners/results.py (JSONL records)
           → runners/pipeline.py / checkpoint.py (resume)
           → statistics/prepare.py (parquet dataset)
           → statistics/* (analysis exports)
```

## Provider layer

| Module | Purpose |
|--------|---------|
| `models/ollama_provider.py` | Local Ollama; v1.1 provenance fields |
| `models/ollama_client.py` | HTTP client; `think` flag resolution |
| `models/openai_provider.py` | API providers |
| `models/types.py` | `ModelRequest` / `ModelResponse` schema |

## Benchmark adapters

| Module | Status |
|--------|--------|
| `benchmarks/humaneval_plus.py` | **AUTHORITATIVE** for Paper 1 (164 tasks) |
| `benchmarks/mbpp.py` | Loader present; MBPP+ path partial (no full Paper 1 campaign) |

## Paper 1 evidence tiers

| Tier | Location | Classification |
|------|----------|----------------|
| v1.0 confirmatory freeze | `artifacts/paper1/frozen/` | AUTHORITATIVE |
| Analysis reproduction | `artifacts/paper1/scripts/` | AUTHORITATIVE |
| qwen3 Arms A/B outputs | `outputs/` (gitignored) + `scripts/` | DIAGNOSTIC |
| Manuscript | sibling `../paper1/` | External to this repo |

## Version lines

- **v1.0.0** (tag + Zenodo): historical confirmatory behavior; frozen parquet.
- **v1.1.0** (`main`): reasoning metadata, raw responses, `budget_exhausted` status.

## Dead / superseded (retained for audit)

- `statistics/gtheory.py`, legacy D-study artifacts: **SUPERSEDED** (not cited in Paper 1 primary path)
- Root `experiments/` trees: **HISTORICAL** (gitignored local dumps)
