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

# Robustness tests (requires trained model)
python scripts/robustness_tests.py --model models/v8/model.zip --all
python scripts/robustness_tests.py --model models/v8/model.zip --monte-carlo 1000
python scripts/robustness_tests.py --model models/v8/model.zip --stress-test --crash-pct -0.30

# Paper trading
python scripts/paper_trade_v7.py
```

## Architecture

```
config/          Config system (dataclasses + YAML)
  config.py        PloutosConfig dataclass - used by older scripts
  settings.py      Auto-detects machine role (WSL/Proxmox/Dev), sets globals
  tickers.py       Ticker lists
  *.yaml           Training configs (V6, V7, V8)
core/            Core ML & environment logic
  universal_environment_v8_lstm.py  Current V8 gym environment
  universal_environment_v6_better_timing.py  Legacy V6 env
  data_fetcher.py  Yahoo Finance data fetching
  macro_data.py    VIX/TNX/DXY macro indicators
  ensemble.py      Multi-model ensemble voting
  data_pipeline.py Feature engineering pipeline
  sp500_scanner.py S&P 500 sector scanning
  risk_manager.py  Risk management logic
  transaction_costs.py  Realistic cost modeling
trading/         Broker integrations
  broker_interface.py  Abstract broker interface
  broker_factory.py    Factory: eToro (default) or Alpaca
  alpaca_client.py     Alpaca API client
  etoro_client.py      eToro API client
  portfolio.py         Portfolio tracking
  stop_loss_manager.py Stop loss logic
training/        Training scripts
  train_walk_forward.py  V8 walk-forward training (main)
  train_v7_sp500_sectors.py  V7 sector-based training
scripts/         Standalone scripts (backtest, optimize, paper trade)
tests/           pytest test suite
dashboard/       Flask web dashboard
notifications/   Discord alerts
database/        PostgreSQL integration
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

- **Two config systems**: `config/config.py` (PloutosConfig dataclass, used by older V6/V7 scripts) vs `config/*.yaml` (used by V8 walk-forward). They have different field names and structures. V8 scripts load YAML directly.
- **Auto-detection in settings.py**: `config/settings.py` detects the machine at import time (WSL -> TRAINING role with GPU + 64 envs, Proxmox -> PRODUCTION, else -> DEV with 4 envs). This runs on import, printing role info to stdout.
- **Data caching**: `data_cache/` is gitignored. Yahoo Finance data gets cached locally to avoid re-downloading.
- **Models gitignored**: `models/` dir is in `.gitignore`. Trained models must be managed separately.
- **Wandb disabled by default**: Set `wandb.enabled: true` in YAML config or configure `WANDB_CONFIG` in settings.py.

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
