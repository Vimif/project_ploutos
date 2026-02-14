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

# Training (V8 walk-forward)
python training/train_walk_forward.py --config config/training_config_v8.yaml
python training/train_walk_forward.py --config config/training_config_v8.yaml --recurrent    # LSTM
python training/train_walk_forward.py --config config/training_config_v8.yaml --ensemble 3   # Ensemble

# Hyperparameter optimization
python scripts/optimize_hyperparams.py --config config/training_config_v8.yaml --n-trials 50
python scripts/optimize_hyperparams.py --config config/training_config_v8.yaml --n-trials 50 --n-jobs 4  # parallel

# Cloud training (RTX 3090 / A5000 — 32 envs, bigger batches)
python training/train_walk_forward.py --config config/training_config_v8_cloud.yaml
python scripts/optimize_hyperparams.py --config config/training_config_v8.yaml --n-trials 50 --n-jobs 4

# Robustness tests (requires trained model)
python scripts/robustness_tests.py --model models/v8/model.zip --all
python scripts/robustness_tests.py --model models/v8/model.zip --monte-carlo 1000
python scripts/robustness_tests.py --model models/v8/model.zip --stress-test --crash-pct -0.30

# Paper trading
python scripts/paper_trade_v7.py
```

## Architecture

```
config/          YAML configs + settings
  settings.py      Auto-detects machine role (WSL/Proxmox/Dev), sets globals
  tickers.py       Ticker lists
  training_config_v8.yaml        Standard training config
  training_config_v8_cloud.yaml  GPU cloud (32 envs, bigger batches)
  training_config_v7_sp500.yaml  Legacy V7 sector config
core/            Core ML & environment logic
  universal_environment_v8_lstm.py  Current V8 gym environment (1318 dims)
  universal_environment_v6_better_timing.py  Legacy V6 env (used by V7/tests)
  advanced_features_v2.py  60+ technical features (support/resistance, RSI, etc.)
  data_fetcher.py  Yahoo Finance data fetching
  macro_data.py    VIX/TNX/DXY macro indicators (V8 only)
  ensemble.py      Multi-model ensemble voting
  data_pipeline.py Feature engineering pipeline + train/val/test splitting
  transaction_costs.py  Realistic slippage/spread/commission model
training/        Training scripts
  train_walk_forward.py  V8 walk-forward training (main entry point)
  train_v7_sp500_sectors.py  V7 sector-based training (legacy)
scripts/         CLI tools (optimize, robustness tests, paper trade)
trading/         Broker integrations (eToro, Alpaca)
tests/           pytest test suite
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
- **Two training configs**: `training_config_v8.yaml` (dev, 8 envs) vs `training_config_v8_cloud.yaml` (GPU cloud, 32 envs). Use cloud config on RunPod/Lambda.
- **Auto-detection in settings.py**: `config/settings.py` detects machine at import time (WSL -> TRAINING, Proxmox -> PRODUCTION, else -> DEV). Prints role to stdout. Still used by `trading/portfolio.py` and `core/utils.py`.
- **SubprocVecEnv vs DummyVecEnv**: PPO uses SubprocVecEnv (parallel). RecurrentPPO requires DummyVecEnv (sequential). This is handled in `train_walk_forward.py`.
- **Data caching**: `data_cache/` is gitignored. Yahoo Finance data gets cached locally.
- **Models gitignored**: `models/` dir is in `.gitignore`. Trained models must be managed separately.
- **Wandb disabled by default**: Set `wandb.enabled: true` in YAML config.

# Karpathy-inspired coding guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" -> "Write tests for invalid inputs, then make them pass"
- "Fix the bug" -> "Write a test that reproduces it, then make it pass"
- "Refactor X" -> "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
