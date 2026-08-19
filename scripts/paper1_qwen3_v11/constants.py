from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # caliper/
PAPER1_ROOT = REPO_ROOT.parent / "paper1"
ANALYSIS_DIR = PAPER1_ROOT / "paper1_analysis_v11"

FROZEN_PARQUET = REPO_ROOT / "artifacts" / "paper1" / "frozen" / "statistical_dataset.parquet"
EXPECTED_FROZEN_SHA256 = (
    "95209fff2f742d59b52aa5cf5616f1395ef7fc01fd87fe025c485577bca0d1c9"
)

EXPECTED_CELLS = 6560
EXPECTED_TOTAL_PANEL = 39360
QWEN3_MODEL = "qwen3_32b"
RANDOM_SEED = 20260404
BOOTSTRAP_REPS = 2000
DPI = 300

ARM_CONFIGS = {
    "a": REPO_ROOT / "configs" / "paper1" / "paper1_confirmatory_humaneval_qwen3_v11_arm_a.yaml",
    "b": REPO_ROOT / "configs" / "paper1" / "paper1_confirmatory_humaneval_qwen3_v11_arm_b.yaml",
}

ARM_OUTPUT_ROOTS = {
    "a": REPO_ROOT / "outputs" / "paper1_confirmatory_humaneval_qwen3_v11_arm_a",
    "b": REPO_ROOT / "outputs" / "paper1_confirmatory_humaneval_qwen3_v11_arm_b",
}

ARM_PROTOCOL = {
    "a": {"think": False, "num_predict": 1024, "label": "Arm A (think=false, num_predict=1024)"},
    "b": {"think": True, "num_predict": 4096, "label": "Arm B (think=true, num_predict=4096)"},
}

REQUIRED_PROVIDER_FIELDS = (
    "done_reason",
    "eval_count",
    "prompt_eval_count",
    "thinking_length",
    "thinking_sha256",
)

FACTOR_KEYS = ("task_id", "prompt_id", "temperature", "run_index")
