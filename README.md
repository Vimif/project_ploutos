# Ploutos Trading V9.1

Experimental trading research system built around walk-forward evaluation,
profitability audits, league batches, and an eToro-focused demo workflow.

This repository is still a research and paper-trading project. It is not a
production trading platform and should not be used with real money without
independent validation.

## Supported Golden Path

The supported runtime path is:

1. `training/train.py` for walk-forward training
2. `scripts/run_pipeline.py` for the main train -> robustness flow
3. `scripts/compare_strategies.py` for family bake-offs
4. `scripts/profitability_audit.py` for profit and risk review
5. `scripts/run_league_batch.py` for challenge -> gold -> demo batch evaluation
6. `scripts/paper_trade.py` for the eToro-first demo loop
7. `dashboard/app.py` for local read-only monitoring

Everything outside that path should be treated as optional integration,
compatibility code, or archived history.

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/Vimif/project_ploutos
cd project_ploutos

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

pip install -e ".[dev,training,web]"
cp .env.example .env
```

Fill in `.env` only if you need broker integrations.

## Main Commands

```bash
# Tests
pytest
pytest tests/test_trading_env_v8.py

# Lint and format
black --check .
ruff check .

# Full pipeline
python scripts/run_pipeline.py --config config/config.yaml --auto-scale --ensemble 3

# Training
python training/train.py --config config/config.yaml --auto-scale
python training/train.py --config config/config.yaml --recurrent --auto-scale
python training/train.py --config config/config.yaml --ensemble 3 --auto-scale

# Hyperparameter search
python scripts/optimize_hyperparams.py --config config/config.yaml --n-trials 50 --auto-scale

# Robustness
python scripts/robustness_tests.py --model models/<fold>/model.zip --all --auto-scale

# Strategy comparison and promotion gates
python scripts/compare_strategies.py --config config/config.yaml
python scripts/profitability_audit.py --walk-forward models/<run_dir>
python scripts/run_league_batch.py --config config/config.yaml

# Demo trading and dashboard
python scripts/paper_trade.py --mode etoro --config config/config.yaml
python dashboard/app.py

# Repo audit and cleanup
python scripts/audit_repo.py
python scripts/housekeeping.py
python scripts/housekeeping.py --apply
python scripts/archive_artifacts.py
python scripts/archive_artifacts.py --apply
```

## Repository Layout

```text
config/        Main runtime config, schema, hardware scaling, tickers
core/          Environment, rewards, features, artifacts, shared memory
training/      Walk-forward training and league evaluation logic
trading/       Broker integrations and live execution
dashboard/     Local Flask demo dashboard
scripts/       Supported CLI entry points
tests/         Pytest coverage for the supported path
legacy/        Archived code, configs, and ops helpers
docs/          Current documentation
docs/archive/  Historical notes kept for reference only
```

## Optional Compatibility Tools

These files are still valid, but they are not required for the golden path:

- `scripts/backtest_ultimate.py` for older artifact inspection and compatibility
- `config/training_config_v8_cloud.yaml` for manual hardware overrides when
  `--auto-scale` is not the desired path

## Legacy Boundary

Older validation helpers, V7 configs, and the historical Grafana or Streamlit
ops stack have been isolated under `legacy/`. Archived docs live under
`docs/archive/`.

If a file is under `legacy/`, it is not part of the supported V9.1 contract.

## Current Constraints

- The full test suite depends on optional training dependencies such as `polars`.
- `models/` is gitignored; trained artifacts should be managed outside the repo.
- Some supported scripts still keep compatibility code for older model metadata.
- The demo dashboard is read-only and file-backed; it does not offer manual
  execution actions in V1.

## Documentation

- Roadmap: [docs/ROADMAP.md](docs/ROADMAP.md)
- Architecture: [docs/ARCHITECTURE_V9.md](docs/ARCHITECTURE_V9.md)
- Monitoring: [docs/MONITORING.md](docs/MONITORING.md)
- Monitoring quickstart: [docs/QUICKSTART_MONITORING.md](docs/QUICKSTART_MONITORING.md)
- Developer knowledge: [docs/DEV_KNOWLEDGE.md](docs/DEV_KNOWLEDGE.md)
- RunPod guide: [docs/RUNPOD_GUIDE.md](docs/RUNPOD_GUIDE.md)
- Coding guidelines: [docs/coding_guidelines.md](docs/coding_guidelines.md)
- Archived docs: [docs/archive/README.md](docs/archive/README.md)

## License

MIT
