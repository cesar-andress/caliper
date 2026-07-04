# Ollama local models for CALIPER

CALIPER can run Paper 1 experiments against **local models served by Ollama**
using the `ollama` provider type. No cloud API keys are required.

## Prerequisites

1. Install [Ollama](https://ollama.com/download)
2. Start the Ollama daemon (usually automatic after install)
3. Pull the models you plan to evaluate

### Check Ollama is running

```bash
curl http://localhost:11434/api/tags
```

Or via CALIPER:

```bash
caliper ollama list
```

If Ollama is not running you should see a clear connection error.

### List local models

```bash
caliper ollama list
caliper ollama list --base-url http://localhost:11434
```

This calls `GET /api/tags` and prints installed model names.

### Pull recommended models

```bash
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:14b
ollama pull deepseek-coder-v2:lite
ollama pull llama3.1:8b
```

## RTX 4090 recommendations (24 GB VRAM)

| Model | Tag | Notes |
|-------|-----|-------|
| Qwen2.5 Coder 7B | `qwen2.5-coder:7b` | Fast smoke tests, low VRAM |
| Qwen2.5 Coder 14B | `qwen2.5-coder:14b` | Good quality/speed trade-off |
| DeepSeek Coder V2 Lite | `deepseek-coder-v2:lite` | Strong coding model, moderate VRAM |
| Llama 3.1 8B | `llama3.1:8b` | General baseline |
| Qwen2.5 Coder 32B | `qwen2.5-coder:32b` | Quantized variants may fit; expect slower runs |
| Qwen3 32B | `qwen3:32b` | Large; use quantized build or expect CPU offload |

Start with **`qwen2.5-coder:7b`** for smoke tests before launching the full pilot.

## Provider configuration

```yaml
providers:
  ollama_local:
    provider_type: ollama
    base_url: http://localhost:11434
    timeout_seconds: 300
```

Equivalent form:

```yaml
providers:
  ollama_local:
    type: ollama
    base_url: http://localhost:11434
    extra:
      timeout_seconds: 300
```

CALIPER uses `POST /api/generate` with:

- `prompt`, `model`, `temperature`, `top_p`
- `num_predict` ← mapped from `max_tokens`
- `seed` when set in decoding config
- `stop` sequences when configured

## Smoke experiment (12 cells)

Validates the pipeline with one local model before larger runs.

```bash
caliper validate --config configs/paper1/ollama_smoke.yaml
caliper plan     --config configs/paper1/ollama_smoke.yaml
caliper run      --config configs/paper1/ollama_smoke.yaml
```

Design: **1 model × 3 tasks × 2 prompts × 1 temperature × 2 runs = 12 cells**

## Paper 1 Ollama pilot (6000 cells)

Full local-model factorial pilot for variance decomposition:

```bash
caliper validate --config configs/paper1/ollama_pilot_variance.yaml
caliper plan     --config configs/paper1/ollama_pilot_variance.yaml
caliper run      --config configs/paper1/ollama_pilot_variance.yaml
```

Design: **6 models × 20 tasks × 5 prompts × 2 temperatures × 5 runs = 6000 cells**

Ensure all six models are pulled before starting. The run is resumable via checkpoints.

## Determinism and seed caveats

- Ollama **may ignore or partially honor** `seed` depending on model, quantisation, and backend.
- GPU scheduling and batching introduce **residual nondeterminism** even with `temperature: 0.0`.
- Treat repeated runs as **stochastic replicates**, not bit-identical reproductions.
- For strict regression testing, use the **mock pilot** (`pilot_variance_decomposition.yaml`).

## Performance notes

- Set `timeout_seconds: 300` (or higher) for large models and long prompts.
- Keep `execution.parallel_workers: 1` unless you run multiple Ollama instances.
- Code tasks use `max_tokens: 256` by default in Paper 1 configs.
- Failed cells (missing model, timeout) are recorded without aborting the experiment.

## Troubleshooting

| Symptom | Action |
|---------|--------|
| Connection refused | Start Ollama: `ollama serve` |
| Model not found | `ollama pull <model>` |
| Timeout | Increase `timeout_seconds` or use a smaller quant |
| CUDA OOM | Use smaller model or quant (e.g. `:7b` instead of `:32b`) |

## Suggested first command

After pulling `qwen2.5-coder:7b`:

```bash
caliper run --config configs/paper1/ollama_smoke.yaml
```
