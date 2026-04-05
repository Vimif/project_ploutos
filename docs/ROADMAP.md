# Ploutos Roadmap

This roadmap focuses on the supported V9.1 workflow, not on every historical
experiment left in the repository.

## Current Priorities

1. Stabilize the golden path

- Keep one supported training entry point: `training/train.py`
- Keep one supported end-to-end pipeline: `scripts/run_pipeline.py`
- Keep one supported robustness path: `scripts/robustness_tests.py`
- Keep one supported paper-trading flow: `scripts/paper_trade.py`

2. Reduce workflow drift

- Keep README, packaging, CI, and runtime commands aligned
- Make config validation fail on typos and impossible combinations
- Remove silent CI fallbacks and duplicated configuration
- Add audits for version drift and broken entry points

3. Strengthen evidence

- Add smoke tests for critical workflows
- Require model metadata and artifact compatibility checks
- Re-run robustness and paper-trading flows on real trained artifacts
- Replace historical claims with reproducible checks

## Completed Foundations

- Walk-forward training
- RecurrentPPO support
- Ensemble training
- Macro data integration
- Shared-memory support
- Polars-based feature engineering
- Basic CI and repository auditing
- Stricter YAML config validation

## Important Reality Checks

- Stress tests must validate behavior on recomputed post-shock features, not on
  stale precomputed indicators.
- Recurrent models must be evaluated with recurrent state preserved.
- The project does not currently implement a true supported short-selling
  workflow in the main environment.
- Legacy V6/V7 compatibility code still exists and should be treated as
  migration debt, not as proof of current support quality.

## Next Cleanup Phases

### Phase 1: Runtime Honesty

- Remove remaining legacy naming from supported scripts
- Simplify docs that overstate maturity or performance
- Add more audit checks for doc/code drift

### Phase 2: Testing the Critical Path

- Add lightweight smoke coverage for train, pipeline, robustness, and paper trade
- Separate optional heavy tests from fast default validation
- Make dependency expectations explicit in CI and docs

### Phase 3: Legacy Isolation

- Move unsupported historical helpers behind clearer compatibility boundaries
- Reduce direct imports from `legacy/` in supported scripts
- Document which model artifact formats are still intentionally supported

Status:
- legacy validation and monitoring assets are isolated under `legacy/`
- the mainline docs describe the supported Flask demo dashboard instead of the
  historical Grafana or Streamlit stack

### Phase 4: Research Features

- Market regime detection
- Transformer-based experiments
- Better monitoring and alerting for paper trading

## Non-Goals For Cleanup

- Rebranding research metrics as production evidence
- Keeping stale version labels for marketing continuity
- Supporting every historical artifact format equally well forever
