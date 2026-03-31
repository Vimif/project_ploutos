# Release Notes V9

This file is kept as a historical summary of the V9 and V9.1 transition.
It is not the canonical source of truth for the current repository state.
For the supported architecture and current cleanup direction, prefer:

- `README.md`
- `docs/ARCHITECTURE_V9.md`
- `docs/ROADMAP.md`

## V9 Highlights

- Introduced Polars-based feature engineering
- Introduced shared-memory support for training workers
- Consolidated the main training flow around the V9 environment
- Expanded the script layer for pipeline, robustness, and paper-trading workflows

## V9.1 Highlights

- Broke out environment responsibilities into smaller components
- Added stricter config validation
- Fixed several evaluation and workflow consistency issues
- Improved repository auditing and packaging alignment

## What Changed Since These Notes Were First Written

Some older V9/V9.1 notes contained fixed benchmark numbers, exact test counts,
and broad maturity claims. Those details drift too quickly and should now be
treated as historical context only, not as current guarantees.

## Current Interpretation

- Performance claims should be re-benchmarked before reuse.
- Test-suite size is less important than whether the critical workflows are
  actually runnable in the current environment.
- Legacy compatibility paths still exist and should not be confused with fully
  supported primary workflows.
