# Paper 1 HumanEval+ protocol comparison

- Subset config: `configs/paper1/confirmatory_humaneval.yaml`
- Full config: `configs/paper1/confirmatory_humaneval_full.yaml`
- Result: **PASS**

## Task coverage

| Study | Task slots | Unique benchmark task IDs | Expected cells |
|-------|------------|---------------------------|----------------|
| 40-task confirmatory | 40 | 40 | 9600 |
| 164-task extension | 164 | 164 | 39360 |

## Expected differences

- experiment_id differs by design
- description differs by design
- output.directory differs by design
- task count: 40 vs 164

## Protocol dimensions checked

- Model set
- Prompt family templates
- Temperature levels
- Run count
- Primary and secondary evaluation metrics
- Provider configuration
- Sandbox execution settings (timeout, memory)
- Random seed policy
- Decoding parameters
- Execution settings (shuffle, workers)
- Logging settings

Only task coverage differs between the two configurations.
