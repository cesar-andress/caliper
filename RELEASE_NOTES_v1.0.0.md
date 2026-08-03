# CALIPER v1.0.0 — Release notes

**Date:** 2026-08-03  
**Git tag:** `v1.0.0`  
**Repository:** https://github.com/cesar-andress/caliper  

## Summary

First public research release of **CALIPER** (*Comparative Analysis of LLM Inference Perturbation, Evaluation, and Ranking*), the executable artifact accompanying Paper 1 on variance-aware factorial evaluation of code LLMs (HumanEval+ confirmatory protocol).

## Authors

| Author | ORCID | Role |
|--------|-------|------|
| César Andrés | [0009-0001-8968-3404](https://orcid.org/0009-0001-8968-3404) | Corresponding |
| David Martín-Moncunill | [0000-0003-2422-9005](https://orcid.org/0000-0003-2422-9005) | Co-author |
| José Manuel Baños | [0009-0004-9971-7390](https://orcid.org/0009-0004-9971-7390) | Co-author |

Affiliation: CRIA-BDHS Research Group, Higher Polytechnic School of Technology and Science, Universidad Camilo José Cela, Madrid, Spain.

## Highlights

- Declarative factorial experiment runner (models × tasks × prompts × temperatures × runs)
- Paper 1 analysis stack: variance decomposition, power simulation, robustness/GLMM, task sampling, design guidance
- Paper 2 ranking-fragility modules
- Example configs under `configs/examples/` and Paper 1 configs under `configs/paper1/`
- MIT license

## What is not included

- Frozen HumanEval+ cell-level experiment outputs (regenerate locally; `experiments/` is gitignored)
- Manuscript LaTeX sources (maintained in the companion paper workspace)

## Citation

Cite the Zenodo archive:

**DOI:** [https://doi.org/10.5281/zenodo.21780089](https://doi.org/10.5281/zenodo.21780089)

See [`CITATION.cff`](CITATION.cff). Development repository: https://github.com/cesar-andress/caliper

## Zenodo

**Published release title:** CALIPER v1.0.0  
**Version DOI:** [10.5281/zenodo.21780089](https://doi.org/10.5281/zenodo.21780089)  
Creators and ORCIDs: see [`.zenodo.json`](.zenodo.json) and [`docs/author_identity.md`](docs/author_identity.md).
