#!/usr/bin/env python3
"""Archived validation helper placeholder.

The original validation pipeline targeted older V7 or V8 workflows and no
longer matches the supported V9.1 runtime or artifact contracts.

This file is kept only to document that the helper existed and has been
retired from the mainline repo.
"""

from __future__ import annotations


def main() -> int:
    print(
        "legacy/scripts/validate_pipeline.py is archived historical material. "
        "Use the supported V9.1 path instead: run_pipeline.py, "
        "compare_strategies.py, profitability_audit.py, and run_league_batch.py."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
