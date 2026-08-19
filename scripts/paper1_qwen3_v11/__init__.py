"""Paper1 qwen3 v1.1 corrective-experiment scientific pipeline.

Do not emit publication numbers from partial arms.
Final analysis: ``python scripts/run_qwen3_v11_final_analysis.py``
"""

__all__ = ["EXPECTED_CELLS", "QWEN3_MODEL"]

EXPECTED_CELLS = 6560
QWEN3_MODEL = "qwen3_32b"
