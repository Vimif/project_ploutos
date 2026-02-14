# Ploutos - RL Trading System

Reinforcement Learning trading bot (PPO/RecurrentPPO) with walk-forward training, ensemble learning, and macro data integration. Currently on V8.

Full roadmap: `docs/ROADMAP.md`

## Commands

```bash
# Install
pip install -e ".[dev,training,web]"

# Tests
pytest                           # all tests
pytest tests/test_trading_env.py # single file

# Lint & Format
black --check .                  # check formatting (line-length=100)
ruff check .                     # lint

# Full pipeline (training + robustness, auto-scales to hardware)
python scripts/run_pipeline.py --config config/training_config_v8.yaml --auto-scale --ensemble 3

# Training (V8 walk-forward)
python training/train_walk_forward.py --config config/training_config_v8.yaml --auto-scale
python training/train_walk_forward.py --config config/training_config_v8.yaml --recurrent --auto-scale
python training/train_walk_forward.py --config config/training_config_v8.yaml --ensemble 3 --auto-scale

# Hyperparameter optimization
python scripts/optimize_hyperparams.py --config config/training_config_v8.yaml --n-trials 50 --auto-scale

# Robustness tests (requires trained model, MC parallelized with --auto-scale)
python scripts/robustness_tests.py --model models/<fold>/model.zip --all --auto-scale

# Paper trading
python scripts/paper_trade_v7.py
```

## Architecture

```
config/          YAML configs + settings
  hardware.py      Auto-detect GPU/CPU/RAM, compute optimal params (--auto-scale)
  schema.py        Lightweight YAML config validation (types + value ranges)
  settings.py      Paths (DATA_DIR, LOGS_DIR, TRADES_DIR), broker, WandB
  tickers.py       Ticker lists
  training_config_v8.yaml        Standard training config
  training_config_v8_cloud.yaml  GPU cloud (manual override, --auto-scale preferred)
core/            Core ML & environment logic
  universal_environment_v8_lstm.py  Current V8 gym environment (1318 dims)
  universal_environment_v6_better_timing.py  Legacy V6 env (used by V7/tests)
  advanced_features_v2.py  60+ technical features (support/resistance, RSI, etc.)
  data_fetcher.py  Yahoo Finance data fetching (max_workers configurable)
  macro_data.py    VIX/TNX/DXY macro indicators (V8 only)
  ensemble.py      Multi-model ensemble voting
  data_pipeline.py Feature engineering pipeline + train/val/test splitting
  transaction_costs.py  Realistic slippage/spread/commission model
training/        Training scripts
  train_walk_forward.py  V8 walk-forward training (main entry point)
  train_v7_sp500_sectors.py  V7 sector-based training (legacy)
scripts/         CLI tools
  run_pipeline.py    Full pipeline: training → robustness (single command)
  optimize_hyperparams.py  Optuna hyperparameter search
  robustness_tests.py  Monte Carlo (parallelized) + stress tests
trading/         Broker integrations (eToro, Alpaca)
tests/           pytest test suite (93 tests: V6 env, V8 env, ensemble, features, pipeline, portfolio)
.github/workflows/tests.yml  CI: pytest + black + ruff on push/PR
```

## Environment Setup

Requires Python >= 3.10.

```bash
cp .env.example .env  # then fill in API keys
```

Key `.env` variables:
- `BROKER` - `etoro` (default) or `alpaca`
- `ALPACA_PAPER_API_KEY` / `ALPACA_PAPER_SECRET_KEY` - for paper trading
- `ETORO_*` - eToro API credentials
- `DB_*` - PostgreSQL (optional)

## Code Style

- **black** with `line-length = 100`, `target-version = py310`
- **ruff** with `line-length = 100`, rules: `E, F, W, I, B, UP` (E501 ignored)
- Match existing style in each file

## Gotchas

- **V8 uses YAML configs directly**: `yaml.safe_load()` in training scripts. No dataclass layer.
- **`--auto-scale` replaces cloud config**: Detects GPU/CPU/RAM and overrides n_envs/batch_size/n_steps. The cloud YAML still works for manual overrides but `--auto-scale` is preferred.
- **settings.py is minimal**: Only provides paths (`LOGS_DIR`, `TRADES_DIR`) and broker config. Used by `core/utils.py` and `trading/portfolio.py`. No training config — that's in YAML + `config/hardware.py`.
- **SubprocVecEnv vs DummyVecEnv**: PPO uses SubprocVecEnv (parallel). RecurrentPPO requires DummyVecEnv (sequential). This is handled in `train_walk_forward.py`.
- **Data caching**: `data_cache/` is gitignored. Yahoo Finance data gets cached locally.
- **Models gitignored**: `models/` dir is in `.gitignore`. Trained models must be managed separately.
- **Wandb disabled by default**: Set `wandb.enabled: true` in YAML config.
- **Config validation**: `config/schema.py` validates YAML on load (types, ranges, typo detection). Runs automatically in `train_walk_forward.py`.

## Coding Guidelines

Follow Karpathy-inspired principles: think before coding, simplicity first, surgical changes, goal-driven execution. Full details in `docs/coding_guidelines.md`.
