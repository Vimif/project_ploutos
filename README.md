# Ploutos Trading AI 🚀

**Reinforcement Learning-based Trading Bot for S&P 500**

Ploutos is an advanced trading AI that uses Proximal Policy Optimization (PPO) to learn profitable trading strategies from historical stock data.

---

## Features

### Core AI
- **PPO Algorithm** - State-of-the-art on-policy RL from Stable Baselines3
- **43,000+ Features** - Comprehensive technical analysis per ticker
- **Multi-Asset Trading** - Simultaneous trading across 500+ stocks
- **GPU Accelerated** - CUDA support for fast training

### Risk Management
- **Automatic Stop Loss** - Configurable % threshold (default: -2%)
- **Take Profit** - Lock in gains at configurable % (default: +4%)
- **Sector Diversification** - AI sees portfolio exposure per sector cluster

### Reward System
- **Differential Sharpe Ratio (DSR)** - Optimizes risk-adjusted returns
- **Drawdown Penalty** - Penalizes portfolio drops
- **Sortino Bonus** - Rewards upside with low downside volatility

### Optimizations
- **VecNormalize** - Automatic observation/reward normalization
- **Vectorized Environment** - O(1) observation generation
- **Linear LR Schedule** - Smooth convergence

---

## Project Structure

```
project_ploutos/
├── config/
│   └── training.yaml       # Training hyperparameters
├── data/
│   └── sp500.csv           # Historical S&P 500 data
├── models/
│   └── ploutos_production/ # Saved models
├── scripts/
│   ├── train.py            # Training script
│   ├── backtest.py         # Backtesting script
│   └── download.py         # Data download utility
├── src/ploutos/
│   ├── env/
│   │   ├── environment.py  # Trading environment
│   │   └── rewards.py      # Reward calculator
│   └── features/
│       └── pipeline.py     # Feature engineering
└── logs/                   # Training logs
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Data
```bash
python scripts/download.py --tickers SP500 --period 5y
```

### 3. Train Model
```bash
python scripts/train.py --config config/training.yaml --timesteps 1000000
```

### 4. Backtest
```bash
python scripts/backtest.py --model models/ploutos_production/final_model.zip --data data/sp500.csv
```

---

## Configuration

Key parameters in `config/training.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_envs` | 32 | Parallel environments |
| `learning_rate` | 1e-4 | Initial learning rate |
| `batch_size` | 2048 | Mini-batch size |
| `n_steps` | 2048 | Steps per rollout |
| `gamma` | 0.99 | Discount factor |

---

## Performance

- **Training Speed**: ~50-100 it/s (GPU, 32 envs)
- **Observation Space**: ~43,000 dimensions
- **Action Space**: 503 discrete actions (Buy/Hold/Sell per ticker)

---

## Requirements

- Python 3.11+
- PyTorch 2.0+ (with CUDA for GPU)
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM (recommended)

---

## License

MIT License - See LICENSE file for details.

---

## Author

Ploutos AI Team - January 2026
