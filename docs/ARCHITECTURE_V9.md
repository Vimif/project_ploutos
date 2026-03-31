# Architecture V9.1

This document describes the supported V9.1 architecture at a high level.
It intentionally avoids hard-coded claims that drift quickly, such as exact
test counts or frozen performance numbers.

## Overview

The project is organized around four main layers:

1. Data and features

- `core/data_fetcher.py`
- `core/macro_data.py`
- `core/features.py`

Market data is fetched as pandas DataFrames. Technical features are computed in
`core/features.py`, using Polars internally for the heavy feature-engineering
path and returning pandas output for environment compatibility.

2. Environment and trading logic

- `core/environment.py`
- `core/env_config.py`
- `core/observation_builder.py`
- `core/reward_calculator.py`
- `core/transaction_costs.py`
- `core/constants.py`

`TradingEnv` is the main environment used by the supported training workflow.
It delegates observation assembly and reward logic to smaller components and can
operate either on raw in-memory data or shared-memory-backed data prepared for
parallel training.

3. Training and evaluation

- `training/train.py`
- `core/model_support.py`
- `core/ensemble.py`

Training is driven by walk-forward splits. The supported workflow computes
features per fold to avoid look-ahead bias. Recurrent evaluation helpers live in
`core/model_support.py` so the same recurrence handling can be reused across
training evaluation and robustness scripts.

4. Operational scripts

- `scripts/run_pipeline.py`
- `scripts/optimize_hyperparams.py`
- `scripts/robustness_tests.py`
- `scripts/paper_trade.py`
- `scripts/backtest_ultimate.py`
- `scripts/audit_repo.py`

These scripts orchestrate the main repo workflows. Some of them still include
legacy compatibility logic for older model formats. That compatibility should be
seen as a maintenance boundary, not as a statement that all historical paths are
equally mature.

## Data Flow

```text
download market data
-> split by walk-forward window
-> compute features inside each fold
-> optionally place training data in shared memory
-> initialize TradingEnv
-> train PPO or RecurrentPPO
-> evaluate on held-out data
-> run robustness checks and paper-trading compatibility flows
```

## Configuration

- `config/config.yaml` is the main runtime config
- `config/schema.py` validates structure, types, ranges, and cross-field constraints
- `config/hardware.py` can auto-scale selected training parameters from hardware

The current validation policy is intentionally strict: unknown sections, typoed
keys, and impossible PPO geometry should fail early.

## Shared Memory Path

`core/shared_memory_manager.py` converts numeric data into a shared-memory
representation so multiple training workers can reuse the same underlying arrays
instead of duplicating DataFrames. The raw path and the shared-memory path must
remain feature-selection compatible.

## Legacy Boundary

The repository still contains a `legacy/` folder and some compatibility logic in
supported scripts, especially around older backtest and artifact formats. The
cleanup direction is:

- keep supported V9.1 flows explicit
- isolate historical compatibility code
- avoid importing legacy behavior into new code unless required

## Quality Controls

- pytest suite in `tests/`
- formatting with Black
- linting with Ruff
- type checking with mypy in CI
- repository audit with `scripts/audit_repo.py`

These controls improve confidence, but the most important validation still comes
from running the critical workflows on real artifacts and checking that docs,
config, and runtime behavior stay aligned.
