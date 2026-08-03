#!/usr/bin/env python3
"""Export Paper 1 design guidance placeholders or populated recommendations."""

from __future__ import annotations

import argparse
from pathlib import Path

from caliper.statistics.design_guidance import export_design_guidance


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Paper 1 design guidance outputs")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("experiments/paper1_confirmatory_humaneval_full/paper1_confirmatory_humaneval_full"),
    )
    parser.add_argument("--metric", default="pass_at_1")
    args = parser.parse_args()
    paths = export_design_guidance(args.experiment_dir, metric=args.metric)
    print(f"Design guidance written to {paths.root}")


if __name__ == "__main__":
    main()
