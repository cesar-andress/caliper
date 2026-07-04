# Local Model Inference

CALIPER supports running open-weight models locally on a single GPU (e.g. NVIDIA RTX 4090) through the `local` provider type and `LocalModelProvider`.

Core CALIPER installs **without** GPU dependencies. Install only the backend you need.

## Quick start

1. Copy and edit the example config:

```bash
cp configs/examples/local_model.yaml my_local_run.yaml
# Set providers.local-gpu.extra.model_path to your checkpoint
```

2. Install a backend (pick one):

```bash
# HuggingFace transformers (recommended default)
pip install -e ".[local]"

# GGUF via llama.cpp
pip install -e ".[local-llama-cpp]"

# vLLM (Linux + CUDA; good throughput on one GPU)
pip install -e ".[local-vllm]"

# Optional NVML power/energy logging
pip install -e ".[local-nvml]"

# Everything
pip install -e ".[local-all]"
```

3. Set the model path (YAML or environment):

```bash
export LOCAL_MODEL_PATH=/data/models/my-checkpoint
```

4. Validate and run:

```bash
caliper validate --config my_local_run.yaml
caliper run my_local_run.yaml
```

Use `--dry-run` on the experiment runner to plan without executing cells, or set `CALIPER_PROVIDER_DRY_RUN=1` / `extra.dry_run: true` on the provider to skip inference while exercising the pipeline.

## YAML configuration

```yaml
providers:
  local-gpu:
    type: local
    extra:
      model_path: /path/to/model          # or HuggingFace hub id org/model
      backend: transformers               # transformers | llama_cpp | vllm
      device: cuda:0
      dtype: bfloat16
      quantization: none                  # none | 4bit | 8bit | awq | gptq | gguf
      deterministic: true
      nvml: true
      nvml_device_index: 0
      timeout_seconds: 300

models:
  - id: local-eval
    provider: local-gpu
    model_id: my-local-model              # logical name in results
```

Per-model overrides are supported via `models[].extra.model_path`.

## Backends

### transformers (default)

Best for HuggingFace checkpoints on a single GPU.

```bash
pip install -e ".[local]"
```

| `quantization` | Requires |
|----------------|----------|
| `none` | torch + transformers |
| `4bit`, `8bit` | bitsandbytes (Linux) |
| `awq` | `autoawq` (install separately) |
| `gptq` | `auto-gptq` (install separately) |

### llama.cpp (`llama_cpp`)

Best for **GGUF** files with `llama-cpp-python`:

```bash
pip install -e ".[local-llama-cpp]"
# CUDA build (example):
# CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

Set `model_path` to the `.gguf` file and `backend: llama_cpp`.

Key options: `n_gpu_layers` (-1 = all layers on GPU), `n_ctx`.

### vLLM (`vllm`)

High-throughput serving-style inference on **one GPU** (`tensor_parallel_size: 1`):

```bash
pip install -e ".[local-vllm]"
```

Requires Linux + CUDA. Set `backend: vllm` and tune `gpu_memory_utilization` (default 0.90).

## Metadata logged per request

Each `ModelResponse.raw_metadata` includes:

- **Quantization**: `quantization`, `dtype`, `backend`
- **GPU**: device name, memory total/allocated, compute capability, CUDA version
- **Latency**: `inference_latency_ms` (backend) and `wall_latency_ms`
- **NVML** (optional): `avg_power_watts`, `energy_joules`, sample count

Structured startup logs (`local.provider.start`) record the same GPU and quantization settings.

## Deterministic inference

When `deterministic: true` (default):

- `temperature: 0` uses greedy decoding where supported
- Request `seed` is forwarded to the backend
- PyTorch / transformers seeds are set when a seed is provided

Some GPU kernels remain slightly non-deterministic; document hardware/driver versions in experiment metadata for reproducibility studies.

## RTX 4090 notes

- 24 GB VRAM fits many 7B–13B models in bf16; use `quantization: 4bit` or GGUF for larger models.
- Prefer `dtype: bfloat16` on Ada Lovelace GPUs.
- Enable `nvml: true` with `pip install -e ".[local-nvml]"` to log power draw and estimated energy per call.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `local model_path is required` | Set `model_path` in YAML or `LOCAL_MODEL_PATH` |
| `transformers backend requires torch` | `pip install -e ".[local]"` |
| CUDA OOM | Lower `max_tokens`, use `quantization: 4bit`, or switch to GGUF |
| vLLM import fails on macOS | vLLM is Linux-only; use `transformers` or `llama_cpp` |

See also: [getting-started.md](getting-started.md), [configs/examples/local_model.yaml](../configs/examples/local_model.yaml).
