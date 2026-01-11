# Ploutos Trading Bot V6 - Custom Data Edition

## Overview
This project is an advanced AI trading bot using Reinforcement Learning (PPO).
This version ("Custom Data Edition") is specifically configured to train on a fixed, high-frequency dataset (`SnP_daily_update.csv`) to learn intraday trading strategies.

## Key Features
- **Algorithm**: PPO (Proximal Policy Optimization) with Stable-Baselines3.
- **Data**: Trains on `SnP_daily_update.csv` (1h timeframe).
- **Environment**: Custom Gymnasium environment (`UniversalTradingEnvV6BetterTiming`).
- **Safety**: Gradient clipping, NaN detection, and action space stability wrappers.

## Structure
- `src/ploutos/`: Core package (Environment, Features, Rewards).
- `scripts/`: Entry points (`train.py`, `download.py`).
- `config/`: Configuration files (`training.yaml`).
- `models/`: Saved models and checkpoints.
- `logs/`: Training logs and TensorBoard data.

## Quick Links
- [Installation Guide](INSTALL.md)
- [Training Guide](TRAINING.md)
