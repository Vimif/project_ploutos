# Architecture Overview

## System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                      Training Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────┐   │
│  │  Data    │───▶│  Feature     │───▶│  TradingEnvironment │   │
│  │  Loader  │    │  Pipeline    │    │  (Gymnasium)        │   │
│  └──────────┘    └──────────────┘    └─────────────────────┘   │
│                                               │                 │
│                                               ▼                 │
│                                      ┌──────────────────┐       │
│                                      │  VecNormalize    │       │
│                                      │  (Normalization) │       │
│                                      └──────────────────┘       │
│                                               │                 │
│                                               ▼                 │
│                                      ┌──────────────────┐       │
│                                      │  SubprocVecEnv   │       │
│                                      │  (Parallelism)   │       │
│                                      └──────────────────┘       │
│                                               │                 │
│                                               ▼                 │
│                                      ┌──────────────────┐       │
│                                      │  PPO Agent       │       │
│                                      │  (MlpPolicy)     │       │
│                                      └──────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. TradingEnvironment (`src/ploutos/env/environment.py`)

**Purpose**: Simulates the stock market for RL training.

**Observation Space** (43,271 dimensions):
- 85 technical features × 503 tickers = 42,755
- 503 portfolio position weights
- 3 global metrics (cash%, return, drawdown)
- 10 sector exposure values

**Action Space**: `MultiDiscrete([3] * 503)`
- 0 = Hold, 1 = Buy, 2 = Sell (per ticker)

**Key Methods**:
| Method | Description |
|--------|-------------|
| `reset()` | Initialize episode |
| `step(actions)` | Execute trades, return obs/reward |
| `_check_risk_management()` | Enforce Stop Loss / Take Profit |
| `_get_observation()` | Generate state vector (O(1)) |

---

### 2. FeaturePipeline (`src/ploutos/features/pipeline.py`)

**Purpose**: Generate 85 technical indicators per ticker.

**Feature Categories**:
| Category | Features | Count |
|----------|----------|-------|
| Support/Resistance | Dynamic levels | ~10 |
| Mean Reversion | Z-scores, Bollinger | ~15 |
| Volume Patterns | OBV, VWAP, Volume ratios | ~15 |
| Price Action | Candlestick patterns | ~20 |
| Divergences | RSI/Price divergence | ~10 |
| Momentum | RSI, MACD, Stochastic | ~15 |

---

### 3. AdvancedRewardCalculator (`src/ploutos/env/rewards.py`)

**Purpose**: Compute reward signal for RL agent.

**Formula**:
```
reward = clip(DSR, -2, 2) + dd_penalty + sortino_bonus + trade_bonus + trade_penalty
```

**Components**:
| Component | Logic |
|-----------|-------|
| **DSR** | Differential Sharpe Ratio (online) |
| **Drawdown Penalty** | -exp(dd × 10) × 0.1 if dd > 5% |
| **Sortino Bonus** | step_return / downside_std × 0.5 |
| **Trade Bonus** | +1.0 for winning trades |
| **Trade Penalty** | -0.1 × (trades - 5) if overtrading |

---

### 4. Risk Management

**Stop Loss**: Triggers when Low price ≤ Entry × (1 - 2%)
**Take Profit**: Triggers when High price ≥ Entry × (1 + 4%)

Both are checked in `_check_risk_management()` before agent actions.

---

### 5. Sector Clustering

**Algorithm**: KMeans (10 clusters)
**Input**: Correlation matrix of asset returns
**Output**: `self.ticker_clusters` array mapping each ticker to a sector

The observation includes sector exposure (% of portfolio in each cluster).

---

## Data Flow

```
CSV Data
    │
    ▼
┌─────────────────┐
│ load_data_dict  │  (train.py)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ FeaturePipeline │  85 features/ticker
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Vectorization   │  Convert to NumPy arrays
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Correlation     │  KMeans → 10 sectors
│ Clustering      │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Training Loop   │  PPO.learn()
└─────────────────┘
```

---

## Performance Optimizations

| Optimization | Location | Impact |
|--------------|----------|--------|
| Vectorized Arrays | `environment.py` | O(1) observation |
| Memory Cleanup | `_prepare_features()` | -50% RAM |
| VecNormalize | `train.py` | Stable training |
| SubprocVecEnv | `train.py` | Parallel rollouts |
| GPU Device | `train.py` | Fast matrix ops |
