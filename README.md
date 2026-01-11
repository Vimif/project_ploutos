# Ploutos Trading Bot

Ploutos is an advanced AI trading system powered by Reinforcement Learning (PPO). It is designed to learn intraday trading strategies using high-frequency historical data.

## 🚀 Features
*   **Core**: Proximal Policy Optimization (PPO) via `Stable-Baselines3`.
*   **Environment**: Custom Gymnasium environment `TradingEnvironment` (v6).
*   **Data**: Optimized for S&P 500 hourly data (`data/sp500.csv`).
*   **Architecture**: Modular design with `src/ploutos` package.

## 📂 Structure
```text
project_ploutos/
├── src/ploutos/        # Core package (Env, Features, Logic)
├── scripts/            # Entry points
│   ├── train.py        # Main training script
│   └── download.py     # Data download utility
├── config/             # Configuration files
├── data/               # Datasets (e.g., sp500.csv)
├── models/             # Checkpoints & Final Models
└── logs/               # TensorBoard & Text logs
```

## ⚡ Quick Start
1.  **Install**: Follow [INSTALL.md](INSTALL.md).
2.  **Download Data**: `python scripts/download.py` (or place your CSV in `data/sp500.csv`).
3.  **Train**: See [TRAINING.md](TRAINING.md).
