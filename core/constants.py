# core/constants.py
"""Centralized constants for the Ploutos trading system.

Avoids magic numbers scattered across environment, reward, and training code.
"""

# --- Observation Space ---
OBSERVATION_CLIP_RANGE = 10.0  # VecNormalize obs clipping range

# --- Finance ---
TRADING_DAYS_PER_YEAR = 252  # Standard finance calendar
HOURS_PER_TRADING_DAY = 6.5  # NYSE market hours
STEPS_PER_TRADING_WEEK_DEFAULT = 78  # 5 days * 6.5h * 2 (30min bars) ≈ 65, hourly ≈ 32

# --- DSR (Differential Sharpe Ratio) ---
DSR_VARIANCE_FLOOR = 1e-4  # Min variance to avoid div-by-zero in flat markets

# --- Portfolio ---
PORTFOLIO_HISTORY_WINDOW = 252  # 1-year rolling window for Sharpe calculation
RETURNS_HISTORY_WINDOW = 100  # Steps of returns to keep for rolling stats
MIN_POSITION_THRESHOLD = 1e-6  # Below this, position is considered zero
EQUITY_EPSILON = 1e-8  # Small constant to avoid division by zero

# --- Risk ---
BANKRUPTCY_THRESHOLD = 0.5  # Episode ends if equity < initial_balance * this
MAX_REWARD_CLIP = 10.0  # Clip reward to [-this, +this] for training stability

# --- Transaction Costs ---
DEFAULT_VOL_CEILING = 0.05  # Max volatility for slippage normalization
