#!/usr/bin/env python3
"""Compatibility shim for the archived validation pipeline."""

from __future__ import annotations


def main() -> int:
    print(
        "scripts/validate_pipeline.py has been archived. "
        "Use scripts/run_pipeline.py for the supported runtime path, "
        "scripts/compare_strategies.py for family bake-offs, and "
        "scripts/profitability_audit.py or scripts/run_league_batch.py "
        "for evaluation and promotion decisions."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
