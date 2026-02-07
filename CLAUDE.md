# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ploutos is a Python algorithmic trading system with two modes:

1. **Trading bot** — RL agent (PPO via Stable-Baselines3) for autonomous paper trading (Alpaca)
2. **Advisory assistant** — Multi-source investment advisor with web dashboard (Flask + Plotly.js)

The README and codebase are a mix of French and English.

## Commands

### Install
```bash
python -m venv venv && venv\Scripts\activate  # Windows
pip install -e .
pip install -e ".[dev]"  # includes pytest, black, ruff, mypy
```

### Train
```bash
python scripts/train_v4_optimal.py
python scripts/train_v4_optimal.py --config config/training_config_v6_better_timing.yaml
```

### Validate a model
```bash
python scripts/validate.py models/<model>.zip
```

### Run live trader (paper)
```bash
python scripts/run_trader_v2_simple.py
```

### Monitor production
```bash
python scripts/monitor_production.py --model models/<model>.zip
python scripts/monitor_production.py --model models/<model>.zip --auto-retrain
```

### Web dashboard
```bash
python web/app.py          # Flask on :5000 (original trading dashboard)
python dashboard/app_v2.py # Streamlit alternative
```

### Advisory dashboard
```bash
python scripts/run_advisor.py              # Flask on :5001
python scripts/run_advisor.py --port 8080  # Custom port
```

### Lint & Format
```bash
black .              # format (line-length=100, py310)
ruff check .         # lint (E, F, W, I rules; E501 ignored)
mypy core utils      # type check
```

### Tests
```bash
pytest                          # all tests with coverage
pytest tests/test_foo.py -v     # single file
pytest tests/test_foo.py::test_bar -v  # single test
```

pytest is configured in `pyproject.toml` with coverage for `core` and `utils`.

## Architecture

Three main pipelines:

### Training Pipeline
`scripts/train_v4_optimal.py` → downloads market data (yfinance, 730d 1h bars) → `core/universal_environment_v6_better_timing.py` (Gymnasium env) → `core/advanced_features_v2.py` (60+ technical features) → PPO training via `SubprocVecEnv` (parallel envs) → saves model to `models/`. Logged to W&B and TensorBoard.

### Trading Pipeline
`scripts/run_trader_v2_simple.py` or `trading/live_trader.py` → loads PPO model → fetches live data via `core/data_fetcher.py` (Alpaca → yfinance fallback) → feature extraction → `model.predict()` → `trading/alpaca_client.py` executes orders → `trading/portfolio.py` tracks positions → `core/risk_manager.py` enforces limits (2% per trade, 3% daily loss, 5% max position).

### Monitoring Pipeline
`web/app.py` (Flask) or `dashboard/app_v2.py` (Streamlit) → reads from PostgreSQL (`database/db.py`) or JSON fallback (`logs/trades/trades_YYYY-MM-DD.json`) → `core/self_improvement.py` scores health and suggests retraining.

## Key Modules

| Directory | Purpose |
|-----------|---------|
| `core/` | RL environment, feature engineering, risk management, data fetching, market analysis |
| `trading/` | Alpaca client, portfolio tracking, live/autonomous trader, stop-loss management |
| `training/` | Trainer classes, curriculum learning, ensemble training |
| `config/` | Dataclass configs (`config.py`), hostname-based settings (`settings.py`), ticker universe (`tickers.py`) |
| `database/` | PostgreSQL schema and queries (`db.py`, `schema.sql`) |
| `advisory/` | Investment advisory engine: adapters, LLM explainer, composite scoring |
| `web_advisor/` | Advisory Flask dashboard (port 5001): API routes + Plotly.js frontend |
| `web/` | Flask API dashboard (original trading) |
| `dashboard/` | Streamlit dashboard with analytics |
| `scripts/` | CLI entry points for training, validation, monitoring, diagnostics, infrastructure |

### Advisory Pipeline

`advisory/engine.py` (`AdvisoryEngine.analyze(symbol)`) orchestrates 5 sub-analyzers, each producing a `SubSignal(signal[-1,+1], confidence[0,1])`:

- **TechnicalAdapter** (weight 0.30) — wraps `dashboard/technical_analysis.py` + `core/advanced_features_v2.py`
- **MLAdapter** (weight 0.25) — wraps `trading/brain_trader.py` PPO model predictions
- **SentimentAnalyzer** (weight 0.20) — Finnhub news + VADER scoring (requires `FINNHUB_API_KEY`)
- **StatisticalForecaster** (weight 0.15) — AutoARIMA via `statsforecast`, 5-day forecast with confidence intervals
- **RiskAdapter** (weight 0.10) — wraps `core/risk_manager.py`

Composite score = weighted average (signals × confidences × weights). Mapped to: ACHAT_FORT / ACHAT / NEUTRE / VENTE / VENTE_FORTE. LLM explanation generated via Ollama (Mistral), with template fallback if Ollama is offline.

Frontend: `web_advisor/` — Flask on port 5001 with Plotly.js candlestick charts, RSI/MACD subplots, forecast bands, and score gauges. All UI text in French.

## Important Patterns

- **Graceful degradation**: External dependencies use optional import flags (`ALPACA_AVAILABLE`, `DB_AVAILABLE`). PostgreSQL falls back to JSON files. Alpaca falls back to yfinance.
- **Machine detection**: `config/settings.py` auto-detects hostname to assign role (TRAINING/PRODUCTION/DEV) with different resource profiles (GPU envs, batch sizes).
- **Environment variables**: API keys and DB credentials loaded from `.env` via `python-dotenv`. Key vars: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, DB connection params, `DISCORD_WEBHOOK_URL`, `FINNHUB_API_KEY`.
- **Path management**: Modules insert project root into `sys.path` for cross-package imports.
- **Action space**: `MultiDiscrete([3] * n_assets)` — each asset gets HOLD(0), BUY(1), or SELL(2) per step.
- **Asset universe**: Defined in `config/tickers.py` with sector allocation (Growth 30%, Defensive 40%, Energy 15%, Finance 15%).

## Code Style

- Python 3.10+ required
- **black**: line-length=100, target py310
- **ruff**: line-length=100, selects E/F/W/I, ignores E501
- Logging uses `core/utils.py:setup_logging()` with console + file handlers
