#!/usr/bin/env bash
# Convenience wrapper — prefer explicit configs for research runs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
exec caliper run --config configs/paper1/confirmatory_humaneval_full.yaml "$@"
