#!/usr/bin/env python3
"""CLI for comparing Ploutos strategy families under one protocol."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.strategy_compare import (  # noqa: E402
    DEFAULT_COMPARE_OUTPUT_DIR,
    compare_strategy_families,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare candidate strategy families")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument(
        "--family",
        action="append",
        default=None,
        help="Candidate family to include. Repeat to compare several families.",
    )
    parser.add_argument("--phase2-top-k", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where strategy_leaderboard.json will be written.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(DEFAULT_COMPARE_OUTPUT_DIR / f"compare_{timestamp}")

    leaderboard = compare_strategy_families(
        config_path=args.config,
        output_dir=output_dir,
        candidate_families=args.family,
        phase2_top_k=args.phase2_top_k,
    )

    winner = leaderboard.get("selection", {}).get("winner_family")
    verdict = leaderboard.get("selection", {}).get("winner_verdict")
    print(f"strategy_leaderboard.json saved to {output_dir}")
    if winner:
        print(f"winner={winner} verdict={verdict}")


if __name__ == "__main__":
    main()
