# V9 Specs Archive

This document is preserved as an archive of the V9 design intent.
It is no longer the best source of truth for the live repository.

## Original Direction

The V9 effort aimed to move the project from a fragile research prototype toward
a more scalable workflow with:

- shared memory for multi-process training
- faster feature engineering
- stronger testing and CI
- cleaner configuration and operational scripts

## What Still Matters

- Shared-memory support remains part of the supported training path.
- Faster feature engineering remains important to the workflow.
- CI, config validation, and audit checks are part of the cleanup strategy.

## What Should Be Read With Caution

- exact benchmark multipliers
- exact RAM reduction claims
- exact test counts
- any statement that implies every historical script path is equally maintained

## Current Source Of Truth

Use these files instead when making changes:

- `README.md`
- `docs/ARCHITECTURE_V9.md`
- `docs/ROADMAP.md`
- `config/config.yaml`
- `training/train.py`
