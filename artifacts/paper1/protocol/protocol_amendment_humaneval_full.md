# Documented protocol amendment — HumanEval+ 164-task extension

**Status:** Documented amendment (not an externally timestamped registry preregistration).  
**Date of comparison artifact:** 2026-07-13 (see sibling comparison files).

## Scope

The confirmatory Paper 1 campaign began with a 40-task HumanEval+ subset
(`configs/paper1/confirmatory_humaneval.yaml`, 9,600 cells) and was extended to
the complete 164-task HumanEval+ suite
(`configs/paper1/confirmatory_humaneval_full.yaml`, 39,360 cells).

## What changed

| Dimension | 40-task subset | 164-task extension |
|-----------|----------------|--------------------|
| Task slots | 40 | 164 |
| Expected cells | 9,600 | 39,360 |
| experiment_id / output directory | subset ids | full ids |

## What did not change

Model tags, controlled prompt family and output contract, temperatures
`{0.0, 0.2}`, run count (5), primary metric (`pass_at_1`), sandbox limits,
provider stack, seed policy, and decoding caps.

Evidence: `paper1_humaneval_full_protocol_comparison.md` (Result: **PASS** —
only task coverage differs).

## Implications for claims

- Analyses of the 164-task freeze are **pre-specified / amendment-documented**,
  not externally preregistered unless a registry timestamp is later attached.
- Primary analysis input remains `frozen/statistical_dataset.parquet` in this
  artifact package.
