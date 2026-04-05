# Architecture V9.1

This document describes the supported V9.1 architecture at a high level.
It intentionally avoids hard-coded claims that drift quickly, such as exact
test counts or frozen performance numbers.

## Overview

The project is organized around five main layers.

### 1. Data and features

- `core/data_fetcher.py`
- `core/macro_data.py`
- `core/features.py`
- `core/data_pipeline.py`

Market data is handled as pandas DataFrames. Technical features are computed in
`core/features.py`, using Polars internally for the heavy feature-engineering
path and returning pandas output for environment compatibility.

### 2. Environment and trading logic

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

### 3. Training and evaluation

- `training/train.py`
- `training/strategy_compare.py`
- `training/league.py`
- `core/ensemble.py`
- `core/model_support.py`

Training is driven by walk-forward splits. Features are computed inside each
fold to avoid look-ahead bias. The comparison and league layers sit above the
base trainer so families, horizons, and promotion gates are evaluated under one
protocol.

### 4. Operational scripts

- `scripts/run_pipeline.py`
- `scripts/optimize_hyperparams.py`
- `scripts/robustness_tests.py`
- `scripts/compare_strategies.py`
- `scripts/profitability_audit.py`
- `scripts/run_league_batch.py`
- `scripts/paper_trade.py`
- `scripts/housekeeping.py`
- `scripts/archive_artifacts.py`
- `scripts/audit_repo.py`

These scripts orchestrate the supported repo workflows. `scripts/backtest_ultimate.py`
is still kept for older artifact inspection and model metadata compatibility,
but it should be treated as a compatibility boundary rather than a primary path.

### 5. Demo monitoring

- `dashboard/app.py`
- `dashboard/demo_monitor.py`
- `dashboard/templates/`

The supported monitoring surface is a local Flask dashboard in read-only mode.
It reads canonical session artifacts from `logs/paper_trading/<session_id>/`
and enriches them with broker state, league outputs, and project-learning
artifacts.

## Data Flow

```text
download or load market data
-> split by walk-forward window
-> compute features inside each fold
-> optionally place training data in shared memory
-> initialize TradingEnv
-> train PPO or RecurrentPPO
-> evaluate on held-out data
-> run robustness checks
-> compare strategy families and build league batches
-> validate in the eToro-focused demo loop
-> inspect the session through the local dashboard
```

## Configuration

- `config/config.yaml` is the main runtime config
- `config/schema.py` validates structure, types, ranges, and cross-field constraints
- `config/hardware.py` can auto-scale selected training parameters from hardware

The current validation policy is intentionally strict: unknown sections, typoed
keys, and impossible PPO geometry should fail early.

## Session And Reporting Artifacts

The canonical live and evaluation artifacts are:

- demo session files:
  - `session_meta.json`
  - `events.jsonl`
  - `equity.jsonl`
  - `report.json`
- strategy comparison:
  - `strategy_leaderboard.json`
- league batches:
  - `league_leaderboard.json`
  - `league_audit.json`
  - `demo_followup.json`
  - `decision_review.json`
  - `project_learning.json`

These artifact names are centralized in `core/artifacts.py` so the paper trader,
league logic, and dashboard do not drift independently.

## Shared Memory Path

`core/shared_memory_manager.py` converts numeric data into a shared-memory
representation so multiple training workers can reuse the same underlying arrays
instead of duplicating DataFrames. The raw path and the shared-memory path must
remain feature-selection compatible.

## Legacy Boundary

The repository still contains a `legacy/` folder for archived validation
helpers, V7 configs, and historical monitoring or ops scripts. New code should
not depend on `legacy/` unless artifact compatibility truly requires it.

## Quality Controls

- pytest suite in `tests/`
- formatting with Black
- linting with Ruff
- type checking with mypy in CI
- repository audit with `scripts/audit_repo.py`

These controls improve confidence, but the most important validation still comes
from running the critical workflows on real artifacts and checking that docs,
config, and runtime behavior stay aligned.
