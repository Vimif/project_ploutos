# Ploutos - RL Trading System

Reinforcement Learning trading bot (PPO/RecurrentPPO) with walk-forward training, ensemble learning, and macro data integration. Currently on V9.

Full roadmap: `docs/ROADMAP.md`

## Commands

```bash
# Install
pip install -e ".[dev,training,web]"

# Tests
pytest                           # all tests
pytest tests/test_trading_env_v8.py # single file

# Lint & Format
black --check .                  # check formatting (line-length=100)
ruff check .                     # lint

# Full pipeline (training + robustness, auto-scales to hardware)
python scripts/run_pipeline.py --config config/config.yaml --auto-scale --ensemble 3

# Training (V9 walk-forward)
python training/train.py --config config/config.yaml --auto-scale
python training/train.py --config config/config.yaml --recurrent --auto-scale
python training/train.py --config config/config.yaml --ensemble 3 --auto-scale

# Hyperparameter optimization
python scripts/optimize_hyperparams.py --config config/config.yaml --n-trials 50 --auto-scale

# Robustness tests (requires trained model, MC parallelized with --auto-scale)
python scripts/robustness_tests.py --model models/<fold>/model.zip --all --auto-scale

# Paper trading
python scripts/paper_trade.py
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
  environment.py     V9 gym environment (configurable obs dims via max_features_per_ticker)
  env_config.py      EnvConfig dataclass (structured config for TradingEnv)
  observation_builder.py  Observation vector construction (extracted from env)
  reward_calculator.py    DSR reward with Welford online variance (extracted from env)
  constants.py       Centralized constants (clip ranges, thresholds, finance params)
  exceptions.py      Custom exception hierarchy (PloutosError, ConfigValidationError, etc.)
  features.py        60+ technical features via Polars (support/resistance, RSI, ADX, etc.)
  data_fetcher.py    Yahoo Finance / Alpaca data fetching (max_workers configurable)
  macro_data.py      VIX/TNX/DXY macro indicators
  ensemble.py        Multi-model ensemble voting with confidence filtering + predict_filtered()
  data_pipeline.py   Feature engineering pipeline + train/val/test splitting
  transaction_costs.py  Realistic slippage/spread/commission model (configurable vol_ceiling)
  shared_memory_manager.py  V9 zero-copy shared memory for SubprocVecEnv
training/        Training scripts
  train.py           V9 walk-forward training (main entry point, per-fold feature computation)
  train_v7_sp500_sectors.py  V7 sector-based training (legacy)
scripts/         CLI tools
  run_pipeline.py    Full pipeline: training → robustness (single command)
  optimize_hyperparams.py  Optuna hyperparameter search
  robustness_tests.py  Monte Carlo (parallelized) + stress tests
trading/         Broker integrations (eToro, Alpaca)
tests/           pytest test suite (116 tests: env, ensemble, features, pipeline, transaction costs, reward, config)
.github/workflows/tests.yml  CI: pytest + black + ruff + mypy (Python 3.10/3.11 matrix)
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

- **V9 uses YAML configs directly**: `yaml.safe_load()` in training scripts. `EnvConfig` dataclass available via `TradingEnv.from_config()` but flat kwargs still work.
- **`--auto-scale` replaces cloud config**: Detects GPU/CPU/RAM and overrides n_envs/batch_size/n_steps. The cloud YAML still works for manual overrides but `--auto-scale` is preferred.
- **settings.py is minimal**: Only provides paths (`LOGS_DIR`, `TRADES_DIR`) and broker config. Used by `core/utils.py` and `trading/portfolio.py`. No training config — that's in YAML + `config/hardware.py`.
- **SubprocVecEnv vs DummyVecEnv**: PPO uses SubprocVecEnv (parallel). RecurrentPPO requires DummyVecEnv (sequential). This is handled in `train.py`.
- **Data caching**: `data_cache/` is gitignored. Yahoo Finance data gets cached locally.
- **Models gitignored**: `models/` dir is in `.gitignore`. Trained models must be managed separately.
- **Wandb disabled by default**: Set `wandb.enabled: true` in YAML config.
- **Config validation**: `config/schema.py` validates YAML on load (types, ranges, typo detection, cross-field constraints for training/environment/data/walk_forward/network/checkpoint/eval). Raises `ConfigValidationError`. Runs automatically in `train.py`.
- **Feature computation inside folds**: Features are computed per-fold inside the walk-forward loop to prevent look-ahead bias. Never pre-compute features on the full dataset.
- **Observation space reduction**: `max_features_per_ticker` in config.yaml selects top-N features by variance (default: 30).
- **Dtype filtering**: `_prepare_features()` uses `pd.api.types.is_numeric_dtype()` to detect feature columns. Never use explicit dtype tuples — Polars `cast(pl.Int32)` produces `int32` which would be missed by `(np.float64, np.float32, np.int64)`.
- **SHM vs raw data path parity**: SharedMemory path homogenizes all numeric types to float32. The raw path (test env) keeps original mixed dtypes. Both paths must select the same features — ensured by `is_numeric_dtype()`.
- **Polars index round-trip**: The FeatureEngineer normalizes the index column to `__date_idx` before Polars conversion and restores it after. This handles any index name ('Date', 'Datetime', None, etc.).

## Coding Guidelines

Follow Karpathy-inspired principles: think before coding, simplicity first, surgical changes, goal-driven execution. Full details in `docs/coding_guidelines.md`.
