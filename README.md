# CALIPER

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21780089.svg)](https://doi.org/10.5281/zenodo.21780089)

**C**omparative **A**nalysis of **L**LM **I**nference **P**erturbation, **E**valuation, and **R**anking

CALIPER is a reproducible research artifact for factorial LLM evaluation experiments.
It treats benchmark scores as measurements that depend on models, tasks, prompts,
decoding settings, and stochastic runs, and it records those facets explicitly so
that variance, uncertainty, and ranking stability can be audited.

**Release:** v1.0.0 — Zenodo [DOI 10.5281/zenodo.21780089](https://doi.org/10.5281/zenodo.21780089)  
**Repository:** [github.com/cesar-andress/caliper](https://github.com/cesar-andress/caliper)

**Authors:** César Andrés (corresponding; [ORCID 0009-0001-8968-3404](https://orcid.org/0009-0001-8968-3404)), David Martín-Moncunill ([ORCID 0000-0003-2422-9005](https://orcid.org/0000-0003-2422-9005)), José Manuel Baños ([ORCID 0009-0004-9971-7390](https://orcid.org/0009-0004-9971-7390)).  
**Affiliation:** CRIA-BDHS Research Group, Higher Polytechnic School of Technology and Science, Universidad Camilo José Cela, Madrid, Spain.  
Identity rules: [`docs/author_identity.md`](docs/author_identity.md).

---

## Contribution

- Declarative factorial designs (YAML): models × tasks × prompts × temperatures × runs
- Resumable local/API execution with cell-level provenance
- Paper 1 analysis stack: descriptive variance partitions, binomial GLMM exports, task-sampling / design guidance
- Paper 2 ranking-fragility modules
- Frozen Paper 1 HumanEval+ statistical dataset packaged under [`artifacts/paper1/`](artifacts/paper1/) for independent analysis reproduction

CALIPER is benchmark-agnostic. The Paper 1 empirical validation uses HumanEval+.

---

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/cesar-andress/caliper.git
cd caliper
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras: `.[api]`, `.[local]`, `.[local-llama-cpp]`, `.[local-vllm]`, `.[local-all]` (see [`docs/local-models.md`](docs/local-models.md)).

```bash
cp .env.example .env   # API keys / local paths; never commit .env
```

---

## Quick start

```bash
caliper validate --config configs/examples/basic_experiment.yaml
caliper plan --config configs/examples/basic_experiment.yaml
caliper run configs/examples/example_factorial.yaml --dry-run
caliper run configs/examples/example_factorial.yaml
make test
```

---

## Running experiments

**Mock (no external services):** [`configs/examples/example_factorial.yaml`](configs/examples/example_factorial.yaml)

**API providers:** install `.[api]`, set keys, configure YAML provider blocks (`openai`, `anthropic`, `gemini`).

**Local / Ollama:** see [`docs/ollama.md`](docs/ollama.md) and [`docs/local-models.md`](docs/local-models.md). Paper 1 used locally served Ollama tags on documented GPU hardware.

Each run writes `outputs/<experiment_id>/<run_id>/` with `manifest.json`, append-only `results.jsonl`, and finalized parquet tables. Resume with `--resume`.

---

## Reproducing Paper 1 (analysis)

Paper 1 tables are reproducible from the frozen statistical dataset **without** re-running inference.

```bash
# 1. Verify integrity of the packaged freeze
python artifacts/paper1/scripts/verify_frozen_dataset.py

# 2. Regenerate core descriptive tables (Type-I shares, compliance rates)
python artifacts/paper1/scripts/reproduce_paper1_core_tables.py

# 3. Optional: verify all packaged file digests
cd artifacts/paper1 && sha256sum -c SHA256SUMS
```

Canonical freeze:

- `artifacts/paper1/frozen/statistical_dataset.parquet` — `N=39,360` cells, metric `pass_at_1`
- SHA-256: `95209fff2f742d59b52aa5cf5616f1395ef7fc01fd87fe025c485577bca0d1c9`
- Freeze docs, recovery audit, integrity audit, locked config snapshot, GLMM/variance exports, and the 40→164 task protocol comparison are under [`artifacts/paper1/`](artifacts/paper1/README.md)

Locked experiment config (for execution reproduction): [`configs/paper1/confirmatory_humaneval_full.yaml`](configs/paper1/confirmatory_humaneval_full.yaml).

**Execution reproduction** of the full factorial requires equivalent Ollama model tags, sandbox settings, and hardware. Exact quantization digests were not recorded in freeze metadata; matching served artifacts by tag alone may not be bitwise identical over time.

Manuscript LaTeX lives in the companion paper workspace; this repository is the software + data artifact.

---

## Repository structure

```text
caliper/                 Python package (CLI, runners, statistics, ranking)
configs/examples/        Mock / tutorial YAML
configs/paper1/          Paper 1 confirmatory configs
artifacts/paper1/        Frozen statistical dataset + analysis reproduction bundle
analyses/paper1/         Analysis entry-point scripts
docs/                    Architecture, Ollama, author identity, release notes
tests/
```

Large local campaign directories under `experiments/` remain gitignored. The public archive ships the analysis-critical freeze via `artifacts/paper1/`.

---

## Citation

Cite the Zenodo software release:

```bibtex
@software{andres2026caliper,
  author    = {Andr{\'e}s, C{\'e}sar and Mart{\'i}n-Moncunill, David and Ba{\~n}os, Jos{\'e} Manuel},
  title     = {{CALIPER}: Comparative Analysis of {LLM} Inference Perturbation,
               Evaluation, and Ranking},
  year      = {2026},
  version   = {1.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21780089},
  url       = {https://doi.org/10.5281/zenodo.21780089},
  note      = {Development repository: https://github.com/cesar-andress/caliper}
}
```

Machine-readable metadata: [`CITATION.cff`](CITATION.cff).

---

## License

MIT — see [`LICENSE`](LICENSE). Third-party models and APIs remain under their own terms.

---

## Support

- Getting started: [`docs/getting-started.md`](docs/getting-started.md)
- Architecture: [`docs/architecture.md`](docs/architecture.md)
- Release notes: [`RELEASE_NOTES_v1.0.0.md`](RELEASE_NOTES_v1.0.0.md)
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)
- Issues: https://github.com/cesar-andress/caliper/issues
