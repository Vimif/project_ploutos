---
name: "ploutos-repo"
description: "Repository workflow skill for the Ploutos RL trading system. Use when Codex is working inside project_ploutos on V9 environment logic, walk-forward training, YAML configuration, feature engineering, robustness tooling, paper trading, or broker integrations and needs repo-specific commands, file map, and guardrails."
---

# Ploutos Repo

## Overview

Work inside this repository with the project's own commands, architecture, and failure modes in mind. Use the bundled reference to avoid re-discovering the same repo context on every task.

## Start Here

- Confirm the current working directory is the `project_ploutos` repository. If not, stop and ask.
- Read `references/repo-context.md` before changing training flow, environment logic, configuration, or data pipelines.
- Read `docs/coding_guidelines.md` before larger refactors or when several implementation paths look plausible.
- Keep changes surgical and match the existing style of the touched files.

## Classify the Task

- Inspect `core/environment.py`, `core/observation_builder.py`, `core/reward_calculator.py`, and `core/env_config.py` for environment, reward, or observation changes.
- Inspect `training/train.py`, `config/config.yaml`, and `config/schema.py` for walk-forward training, YAML validation, or auto-scale behavior.
- Inspect `core/features.py`, `core/data_pipeline.py`, `core/data_fetcher.py`, and `core/macro_data.py` for feature engineering or market data work.
- Inspect `scripts/run_pipeline.py`, `scripts/optimize_hyperparams.py`, `scripts/robustness_tests.py`, and `scripts/paper_trade.py` for CLI or pipeline changes.
- Inspect `trading/` for broker integrations and execution behavior.

## Preserve Repo Invariants

- Compute features inside each walk-forward fold. Do not pre-compute features across the full dataset.
- Use `pd.api.types.is_numeric_dtype()` when selecting numeric feature columns so mixed dtypes and Polars casts stay aligned.
- Treat `--auto-scale` as the preferred training path unless the user explicitly asks for manual overrides.
- Preserve parity between the shared-memory observation path and the raw-data test path.
- Assume `models/` and `data_cache/` are disposable local artifacts unless the user says otherwise.

## Verify Narrowly

- Run the smallest focused `pytest` target that covers the change before escalating to a full suite.
- Run `black --check .` and `ruff check .` when formatting or lint fallout is plausible.
- Avoid full training, robustness sweeps, or paper trading unless the task requires them; call out that gap when they are skipped.

## Use References Deliberately

- Read `references/repo-context.md` for commands, architecture, and gotchas.
- Read `docs/ROADMAP.md` only when the request depends on planned direction rather than current implementation.

## Report Clearly

- Report the exact commands you ran and the scope they covered.
- Call out any assumptions around YAML config, walk-forward folds, cached data, or missing models.
- Mention residual risk explicitly when validation stops short of training or robustness execution.
