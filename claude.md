# CLAUDE.md — AI Assistant Guide for Ploutos Trading Bot

## Project Overview

Ploutos is a reinforcement learning-based algorithmic trading bot for US equities (NYSE/NASDAQ). It uses PPO (Proximal Policy Optimization) via stable-baselines3 to learn autonomous BUY/HOLD/SELL strategies from market data and technical indicators.

**Status**: Paper trading only. Not production-ready for real money.

**Language**: Python 3.10+
**License**: MIT

## Repository Structure

```
project_ploutos/
├── config/                 # YAML configs + Python dataclasses
│   ├── config.py           # PloutosConfig dataclass (central config)
│   ├── training_config_v6_better_timing.yaml  # V6 training (15M steps)
│   ├── training_config_v6_extended_50m.yaml   # V6 extended (50M steps)
│   ├── training_config_v7_sp500.yaml          # V7 S&P 500 sectors
│   ├── autonomous_config.yaml
│   └── test_config.yaml
├── core/                   # Core ML and trading logic
│   ├── universal_environment_v6_better_timing.py  # Active Gym env (V6)
│   ├── advanced_features_v2.py    # 60+ feature engineering
│   ├── data_fetcher.py            # Multi-source data: Alpaca → yfinance → Polygon
│   ├── risk_manager.py            # Kelly criterion, position sizing, drawdown
│   ├── market_analyzer.py         # RSI, MACD, Bollinger, trend analysis
│   ├── market_status.py           # NYSE/NASDAQ open/close status
│   ├── sp500_scanner.py           # S&P 500 sector scanner
│   ├── trading_callback.py        # W&B monitoring callback for training
│   ├── transaction_costs.py       # Slippage and market impact modeling
│   ├── self_improvement.py        # Trade analysis (exists but unused)
│   └── utils.py                   # Logging setup, GPU info, cleanup
├── trading/                # Broker integrations and live trading
│   ├── broker_interface.py        # ABC base class for all brokers
│   ├── broker_factory.py          # Factory: create_broker('etoro'|'alpaca')
│   ├── etoro_client.py            # eToro API integration
│   ├── alpaca_client.py           # Alpaca API integration
│   ├── live_trader.py             # Main trading loop
│   ├── brain_trader.py            # Model-based decision engine
│   ├── autonomous_trader.py       # Autonomous trading engine
│   ├── portfolio.py               # Portfolio tracking
│   └── stop_loss_manager.py       # SL/TP management
├── training/               # Model training scripts
│   ├── train_v6_better_timing.py  # V6 training (15M steps, primary)
│   ├── train_v6_extended_50m.py   # V6 extended training (50M steps)
│   ├── train_v7_sp500_sectors.py  # Sector-based training
│   ├── trainer.py                 # Generic trainer class
│   └── curriculum_trainer.py      # Progressive difficulty learning
├── scripts/                # CLI entry points and utilities
│   ├── run_trader_v6.py           # Main trading script (paper/live)
│   ├── backtest_v6.py             # Backtesting V6 models
│   ├── backtest_reliability.py    # Multi-episode reliability tests
│   ├── analyze_why_fails_v6.py    # Timing diagnostic tool
│   ├── validate.py                # Model validation
│   └── monitor_production.py      # Production monitoring
├── tests/                  # Unit tests
│   ├── test_portfolio.py          # Portfolio trading tests
│   └── verify_days_held.py        # Risk manager verification
├── data/models/            # Trained model files (.zip)
├── dashboard/              # Flask analytics dashboard
├── web/                    # Alternative web interface
├── database/               # PostgreSQL schema and utilities
├── notifications/          # Discord notification integration
├── docs/                   # Extended documentation (15 files)
└── runs/                   # Training run artifacts
```

## Build & Development Commands

```bash
# Install (editable mode)
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Install from requirements.txt
pip install -r requirements.txt

# Training-specific dependencies
pip install -r requirements_training.txt
```

## Testing

```bash
# Run all tests with coverage
pytest tests/ -v --cov=core --cov=utils --cov-report=html

# Run a specific test file
pytest tests/test_portfolio.py -v

# Run a specific test
pytest tests/test_portfolio.py::test_buy_success -v
```

Test configuration is in `pyproject.toml` under `[tool.pytest.ini_options]`:
- Test discovery: `tests/` directory, files matching `test_*.py`
- Coverage: `core` and `utils` modules
- Tests mock heavy dependencies (e.g., `torch`) to run without GPU

## Linting & Formatting

```bash
# Format code with Black
black . --line-length 100 --target-version py310

# Lint with Ruff
ruff check .

# Type check with MyPy
mypy .
```

Configuration (all in `pyproject.toml`):
- **Black**: line-length=100, target Python 3.10
- **Ruff**: rules E, F, W, I; ignores E501 (line length handled by Black)
- **Line length**: 100 characters everywhere

## Key Architectural Patterns

### Environment Versioning
- Evolution: V2 → V3 → V4 → V6 (no V5)
- **Active environment**: `UniversalTradingEnvV6BetterTiming` in `core/universal_environment_v6_better_timing.py`
- All new scripts must use V6. Do not create scripts using older environment versions.

### Broker Abstraction
- `BrokerInterface` (ABC) in `trading/broker_interface.py` defines the contract
- `broker_factory.create_broker()` returns the configured broker
- Supported: eToro (primary, default), Alpaca (fallback)
- Default broker set via `BROKER` env var

### Data Layer with Fallback
Priority order for market data:
1. Alpaca API
2. Yahoo Finance (yfinance)
3. Polygon.io

The `UniversalDataFetcher` in `core/data_fetcher.py` handles automatic failover.

### Configuration System
- YAML configs in `config/` for training scenarios
- Dataclass-based Python config in `config/config.py` (`PloutosConfig`)
- Load with `PloutosConfig.from_yaml(path)` or `PloutosConfig.from_json(path)`
- PPO hyperparameters extracted via `config.get_ppo_kwargs()`

### Standard Tickers (legacy)
```python
TICKERS = ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'TSLA',
           'SPY', 'QQQ', 'VOO', 'VTI', 'XLE', 'XLF', 'XLK', 'XLV']
```
Modern approach: dynamic loading from S&P 500 sectors via `core/sp500_scanner.py`.

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Purpose |
|----------|---------|
| `BROKER` | `etoro` (default) or `alpaca` |
| `ETORO_SUBSCRIPTION_KEY` | eToro API subscription key |
| `ETORO_USERNAME` | eToro account username |
| `ETORO_PASSWORD` | eToro account password |
| `ETORO_API_KEY` | eToro API key |
| `ALPACA_PAPER_API_KEY` | Alpaca paper trading API key |
| `ALPACA_PAPER_SECRET_KEY` | Alpaca paper trading secret |
| `ALPACA_LIVE_API_KEY` | Alpaca live trading API key |
| `ALPACA_LIVE_SECRET_KEY` | Alpaca live trading secret |
| `DB_HOST` | PostgreSQL host (default: localhost) |
| `DB_NAME` | Database name (default: ploutos) |
| `DB_USER` | Database user |
| `DB_PASSWORD` | Database password |

Database is optional — the system falls back to JSON logging in `logs/trades/`.

## Common Workflows

### Training a model
```bash
# Primary V6 training (15M steps, 16 parallel envs)
python training/train_v6_better_timing.py

# Extended training (50M steps)
python training/train_v6_extended_50m.py

# S&P 500 sector-based training
python training/train_v7_sp500_sectors.py
```

### Backtesting
```bash
python scripts/backtest_v6.py --model data/models/brain_tech.zip --days 90
python scripts/backtest_reliability.py --model data/models/brain_tech.zip --episodes 5
```

### Paper trading
```bash
python scripts/run_trader_v6.py --paper --broker etoro --model data/models/brain_tech.zip
```

### Diagnostics
```bash
python scripts/analyze_why_fails_v6.py --model data/models/brain_tech.zip
```

### Dashboard
```bash
python dashboard/app.py        # Flask V1
python dashboard/app_v2.py     # Flask V2
```

### Monitoring
```bash
tensorboard --logdir runs/v6_better_timing/ --port 6006
python scripts/monitor_production.py --model data/models/brain_tech.zip --auto-retrain
```

## Coding Conventions

- **Line length**: 100 characters
- **Formatter**: Black
- **Linter**: Ruff (pyflakes, pycodestyle, isort)
- **Python version**: 3.10+ (use modern type hints, dataclasses)
- **Encoding**: Always use UTF-8 for logs/stdout (`sys.stdout.reconfigure(encoding='utf-8')`) for Windows compatibility
- **Config pattern**: Use YAML for training configs, dataclasses for Python config objects
- **Broker pattern**: Extend `BrokerInterface` ABC for new brokers, register in `broker_factory.py`
- **Environment pattern**: Gym environments subclass `gymnasium.Env`, register observation/action spaces
- **Logging**: Use `core/utils.setup_logging()` for consistent log formatting
- **Models**: Saved as `.zip` files (stable-baselines3 format) in `data/models/`
- **Tests**: Use pytest with fixtures; mock heavy deps like `torch` when not needed

## Risk Management Parameters

| Parameter | Value |
|-----------|-------|
| Max portfolio risk | 2% |
| Daily loss limit | 3% (circuit breaker) |
| Max single position | 5% of portfolio |
| Max correlation | 0.7 between positions |
| Position sizing | Kelly criterion |
| Target Sharpe ratio | > 1.5 (validation threshold) |

## Known Issues

1. **BUY timing problem**: V4 bought too late (85% buy-high). V6 targets >50% buy-low via Features V2 (support/resistance, mean reversion, RSI divergences).
2. **No walk-forward validation**: Backtests only cover recent fixed periods.
3. **SL/TP not integrated in live execution**: Risk manager computes them but they're not enforced.
4. **Self-improvement disconnected**: `core/self_improvement.py` exists but is not wired into the trading loop.
5. **Dashboard V2 not connected**: `dashboard/app_v2.py` exists but is not linked to live trading data.

## Important Notes for AI Assistants

- **Paper trading only**: All configs default to `paper_trading: True`. Never enable real-money trading.
- **No CI/CD**: There are no GitHub Actions or CI pipelines. Run tests locally before committing.
- **No Docker**: The project runs directly on the host. Infrastructure setup is via shell scripts in `scripts/`.
- **Credentials**: Never commit `.env` or hardcode API keys. The project already had a security fix for a hardcoded `SECRET_KEY`.
- **Model files**: `.zip` files in `data/models/` are large (up to 68MB). They are gitignored under `models/` but `data/models/` is tracked.
- **French codebase**: Comments, documentation, variable names, and commit messages are often in French. Maintain this convention when modifying existing French-documented code, but English is acceptable for new standalone files.
