# Ploutos Trading V9.1

Experimental reinforcement-learning trading system built around PPO/RecurrentPPO,
walk-forward validation, ensemble training, technical features, and macro data.

This repository is still a research and paper-trading project. It is not a
production trading platform and should not be used with real money without
independent validation.

## What Is In Scope

- Walk-forward training with `training/train.py`
- End-to-end pipeline with `scripts/run_pipeline.py`
- Robustness checks with `scripts/robustness_tests.py`
- Paper trading with `scripts/paper_trade.py`
- Feature engineering with Polars
- Shared-memory support for multi-process training

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

Fill in `.env` only if you need broker or database integrations.

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

# Paper trading
python scripts/paper_trade.py --model models/<fold>/model.zip
```

## Architecture

```text
config/      YAML config, hardware auto-scaling, settings
core/        Environment, features, rewards, observations, shared memory
training/    Walk-forward training entry point
scripts/     Pipeline, optimization, robustness, paper trading, audits
trading/     Broker integrations
tests/       Pytest suite
docs/        Architecture notes and roadmap
legacy/      Older V6/V7 code kept for compatibility and migration
```

Useful entry points:

- `core/environment.py`
- `core/features.py`
- `core/shared_memory_manager.py`
- `training/train.py`
- `scripts/run_pipeline.py`
- `scripts/robustness_tests.py`
- `scripts/paper_trade.py`

## Current Constraints

- The full test suite depends on optional training dependencies such as `polars`.
- `models/` is gitignored; trained artifacts must be managed outside the repo.
- Some scripts still contain legacy compatibility paths for older model formats.
- CI and audit are meant to catch workflow drift, but they are not a substitute for
  running the critical flows on real artifacts.

## Documentation

- Roadmap: [docs/ROADMAP.md](docs/ROADMAP.md)
- Architecture notes: [docs/ARCHITECTURE_V9.md](docs/ARCHITECTURE_V9.md)
- RunPod guide: [docs/RUNPOD_GUIDE.md](docs/RUNPOD_GUIDE.md)
- Coding guidelines: [docs/coding_guidelines.md](docs/coding_guidelines.md)

## Status

Recent cleanup focused on:

- honest config validation
- reproducible workflow auditing
- recurrent-model evaluation fixes
- robustness pipeline fixes
- paper-trading and packaging drift reduction

More cleanup is still needed around legacy compatibility code and documentation
that historically drifted faster than the runtime.

## License

MIT
