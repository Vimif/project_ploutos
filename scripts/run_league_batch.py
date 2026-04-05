#!/usr/bin/env python3
"""Run a complete Ploutos profit/risk league batch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.league import run_league_batch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a versioned profit/risk league batch")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-id", type=str, default=None)
    parser.add_argument("--snapshot-id", type=str, default=None)
    args = parser.parse_args()

    result = run_league_batch(
        config_path=args.config,
        output_dir=args.output_dir,
        batch_id=args.batch_id,
        snapshot_id=args.snapshot_id,
    )

    print(f"League batch saved to {result['output_dir']}")
    print(
        "batch_id={batch_id} snapshot_id={snapshot_id} audit_verdict={audit_verdict} demo_status={demo_status}".format(
            **result
        )
    )


if __name__ == "__main__":
    main()
