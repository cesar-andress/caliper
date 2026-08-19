#!/usr/bin/env python3
"""Entry point: run the full qwen3 v1.1 scientific pipeline.

Usage (from caliper/ repo root):

  .venv/bin/python scripts/run_qwen3_v11_final_analysis.py
  .venv/bin/python scripts/run_qwen3_v11_final_analysis.py --integrity-only
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper1_qwen3_v11.run_final_analysis import main

if __name__ == "__main__":
    raise SystemExit(main())
