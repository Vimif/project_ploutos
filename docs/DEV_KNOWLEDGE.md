# Developer Knowledge

This document collects the current repo-specific lessons that are still useful
for the supported V9.1 workflow.

## 1. Golden Path First

When in doubt, treat this as the supported path:

- `training/train.py`
- `scripts/run_pipeline.py`
- `scripts/robustness_tests.py`
- `scripts/compare_strategies.py`
- `scripts/profitability_audit.py`
- `scripts/run_league_batch.py`
- `scripts/paper_trade.py`
- `dashboard/app.py`

If a change only makes sense for archived helpers, it probably belongs under
`legacy/`.

## 2. Feature Leakage Rules

- compute features inside each walk-forward fold
- never pre-compute features over the full dataset for the supported trainer
- keep the shared-memory path and raw-data test path feature-compatible
- use numeric dtype detection that survives mixed pandas or Polars outputs

These are non-negotiable if you want believable results.

## 3. Configuration Reality

- `config/config.yaml` is the main config
- `config/schema.py` is strict on purpose
- `config/training_config_v8_cloud.yaml` is a manual override path, not the default
- `--auto-scale` is the preferred training path when possible

Do not build new workflows around historical V7 configs.

## 4. Live And Demo State

The supported local telemetry format is:

- `session_meta.json`
- `events.jsonl`
- `equity.jsonl`
- `report.json`

These names are canonicalized in `core/artifacts.py` and are used by:

- `scripts/paper_trade.py`
- `training/league.py`
- `dashboard/demo_monitor.py`

If one of those layers needs a new artifact, update the shared contract first.

## 5. Current Monitoring Truth

The current dashboard is a local Flask app, not the old PostgreSQL or Grafana
stack. It reads the latest demo session, broker snapshot, strategy comparison,
profitability audit, and project-learning context.

Do not reintroduce documentation that claims the primary dashboard runs on old
`/api/status` or `LiveTrader` paths.

## 6. Compatibility Boundary

Some old artifact formats still matter because the supported code can inspect
their metadata. The clearest example is `scripts/backtest_ultimate.py`, which
still provides model metadata helpers used by the paper trader.

That does not make the whole old validation stack a supported workflow.

## 7. Cleanup Rule

If a file is:

- not part of the golden path,
- not used by current code,
- and only exists for historical context,

prefer moving it into `legacy/` or `docs/archive/` instead of letting it stay
in the mainline surface.
