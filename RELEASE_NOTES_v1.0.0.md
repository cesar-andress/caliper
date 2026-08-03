# CALIPER v1.0.0 — Release notes

**Date:** 2026-08-03 (metadata refresh with Paper 1 freeze package)  
**Git tag:** `v1.0.0`  
**Repository:** https://github.com/cesar-andress/caliper  
**Zenodo DOI:** https://doi.org/10.5281/zenodo.21780089  

## Summary

Public research release of **CALIPER** (*Comparative Analysis of LLM Inference Perturbation, Evaluation, and Ranking*), the executable artifact accompanying Paper 1 on measurement and variance in HumanEval+-style function-level code-synthesis evaluation.

## Authors

| Author | ORCID | Role |
|--------|-------|------|
| César Andrés | [0009-0001-8968-3404](https://orcid.org/0009-0001-8968-3404) | Corresponding |
| David Martín-Moncunill | [0000-0003-2422-9005](https://orcid.org/0000-0003-2422-9005) | Co-author |
| José Manuel Baños | [0009-0004-9971-7390](https://orcid.org/0009-0004-9971-7390) | Co-author |

Affiliation: CRIA-BDHS Research Group, Higher Polytechnic School of Technology and Science, Universidad Camilo José Cela, Madrid, Spain.

## Highlights

- Declarative factorial experiment runner (models × tasks × prompts × temperatures × runs)
- Paper 1 analysis stack and Paper 2 ranking-fragility modules
- **Paper 1 frozen statistical dataset** packaged under `artifacts/paper1/` (`N=39,360`, checksum-verified)
- Protocol comparison / documented 40→164 task amendment under `artifacts/paper1/protocol/`
- Example configs under `configs/examples/` and Paper 1 configs under `configs/paper1/`
- MIT license

## Artifact contents for Paper 1 reproduction

See [`artifacts/paper1/README.md`](artifacts/paper1/README.md).

Included for independent analysis reproduction:

- `frozen/statistical_dataset.parquet`
- freeze / integrity / recovery documents
- locked GLMM and variance CSV exports
- verification and core-table reproduction scripts

Not included (by design):

- Full raw `results.jsonl` campaign dumps (large; local `experiments/` tree)
- Model weights / Ollama blobs
- Manuscript LaTeX sources

## Citation

**DOI:** [https://doi.org/10.5281/zenodo.21780089](https://doi.org/10.5281/zenodo.21780089)  
See [`CITATION.cff`](CITATION.cff).
