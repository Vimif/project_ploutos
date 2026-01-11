# Training Guide

## Overview

Ploutos uses **Proximal Policy Optimization (PPO)** from Stable Baselines3 to train a trading agent. The agent learns to maximize risk-adjusted returns (Differential Sharpe Ratio) across 500+ stocks.

---

## Quick Start

```bash
python scripts/train.py --config config/training.yaml --timesteps 1000000
```

---

## Training Parameters

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `config/training.yaml` | Configuration file |
| `--output` | `models/ploutos_production` | Model save directory |
| `--device` | `cuda:0` | Training device |
| `--timesteps` | `50000000` | Total training steps |
| `--seed` | `42` | Random seed |
| `--data` | `data/sp500.csv` | Training data file |

### Configuration File (`training.yaml`)

```yaml
training:
  n_envs: 32           # Parallel environments
  batch_size: 2048     # Mini-batch size
  learning_rate: 1e-4  # Initial LR (uses linear decay)
  n_epochs: 10         # PPO epochs per update
  gamma: 0.99          # Discount factor
  gae_lambda: 0.95     # GAE lambda
  clip_range: 0.2      # PPO clip range
```

---

## Training Pipeline

### 1. Data Loading
- Loads multi-ticker CSV with OHLCV data
- Converts to dictionary format: `{ticker: DataFrame}`

### 2. Feature Engineering
- 85 technical indicators per ticker
- Includes: RSI, MACD, Bollinger Bands, Support/Resistance, Volume patterns

### 3. Environment Setup
- Creates vectorized parallel environments (`SubprocVecEnv`)
- Wraps with `VecNormalize` for observation/reward normalization

### 4. Model Training
- PPO with MLP policy
- Linear learning rate decay
- Checkpoints every 500k steps

---

## Optimizations Applied

| Optimization | Effect |
|--------------|--------|
| **VecNormalize** | Normalizes observations and rewards automatically |
| **Linear LR Schedule** | LR decays from 5e-5 → 0 over training |
| **Gradient Clipping** | `max_grad_norm=0.5` prevents exploding gradients |
| **NaN Detection** | Stops training if numerical instability detected |

---

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir logs/tensorboard
```

Open `http://localhost:6006` in browser.

### Key Metrics to Watch
- `rollout/ep_rew_mean` - Average episode reward (should increase)
- `train/loss` - Overall loss (should decrease)
- `train/policy_gradient_loss` - Policy loss
- `train/value_loss` - Value function loss

---

## Checkpoints

Models are saved to `models/ploutos_production/`:
- `checkpoint_500000_steps.zip` - Every 500k steps
- `final_model.zip` - End of training
- `interrupted_model.zip` - If training is interrupted (Ctrl+C)

---

## Resume Training

To resume from a checkpoint:
```bash
# Not directly supported, but you can load and continue:
python -c "
from stable_baselines3 import PPO
model = PPO.load('models/ploutos_production/checkpoint_500000_steps.zip')
# Continue training...
"
```

---

## Recommended Training Schedule

| Phase | Steps | Purpose |
|-------|-------|---------|
| Test Run | 100k | Verify setup works |
| Short Training | 1M | Quick iteration |
| Full Training | 10M+ | Production model |

---

## Troubleshooting

### Training Very Slow (< 10 it/s)
- Check GPU is being used: Look for "Using cuda:0" in logs
- Reduce `n_envs` if RAM is limiting

### Loss is NaN
- Training will auto-stop after 5 NaN detections
- Try reducing `learning_rate` in config

### Out of Memory
- Reduce `n_envs` (try 16 or 8)
- Reduce `batch_size` (try 1024)

---

## Next Steps

After training, run backtesting:
```bash
python scripts/backtest.py --model models/ploutos_production/final_model.zip --data data/sp500.csv
```

This generates:
- `backtest_equity.png` - Portfolio equity curve
- `backtest_trades_[TICKER].png` - Trade visualization per ticker
