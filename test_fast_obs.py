import time
import numpy as np
from collections import deque
from core.observation_builder import ObservationBuilder
from core.constants import OBSERVATION_CLIP_RANGE

class FastObservationBuilder(ObservationBuilder):
    def build(
        self,
        current_step: int,
        portfolio,
        prices,
        equity,
        balance,
        initial_balance,
        peak_value,
        entry_prices=None,
        portfolio_value_history=None,
    ) -> np.ndarray:
        clip = OBSERVATION_CLIP_RANGE
        obs = np.zeros(self.obs_size, dtype=np.float32)
        idx = 0

        # Technical features per ticker
        for ticker in self.tickers:
            features_array = self.feature_arrays[ticker]
            if current_step >= len(features_array):
                obs[idx : idx + self.n_features] = 0.0
            else:
                obs[idx : idx + self.n_features] = features_array[current_step]
            idx += self.n_features

        # Macro features
        if self.macro_array is not None:
            if current_step < len(self.macro_array):
                obs[idx : idx + self.n_macro_features] = self.macro_array[current_step]
            else:
                obs[idx : idx + self.n_macro_features] = 0.0
            idx += self.n_macro_features

        # Position percentages
        for ticker in self.tickers:
            price = prices.get(ticker, 0.0)
            if price > 0:
                position_value = portfolio.get(ticker, 0.0) * price
                position_pct = position_value / (equity + 1e-8)
            else:
                position_pct = 0.0
            obs[idx] = min(max(position_pct, 0.0), 1.0)
            idx += 1

        # Unrealized PnL per position
        if entry_prices is None:
            entry_prices = {}
        for ticker in self.tickers:
            entry = entry_prices.get(ticker, 0.0)
            qty = portfolio.get(ticker, 0.0)
            price = prices.get(ticker, 0.0)
            if entry > 0 and qty > 0 and price > 0:
                unrealized_pnl = (price - entry) / entry
            else:
                unrealized_pnl = 0.0
            obs[idx] = min(max(unrealized_pnl, -1.0), 5.0)
            idx += 1

        # Portfolio state
        obs[idx] = min(max(balance / (equity + 1e-8), 0.0), 1.0)
        obs[idx+1] = min(max((equity - initial_balance) / initial_balance, -1.0), 5.0)
        obs[idx+2] = min(max((peak_value - equity) / (peak_value + 1e-8), 0.0), 1.0)
        idx += 3

        # Recent portfolio returns (1-step, 5-step, 20-step)
        hist = list(portfolio_value_history) if portfolio_value_history else []

        def _recent_return(lookback):
            if len(hist) > lookback and hist[-lookback - 1] > 0:
                return (hist[-1] - hist[-lookback - 1]) / hist[-lookback - 1]
            return 0.0

        obs[idx] = min(max(_recent_return(1), -0.5), 0.5)
        obs[idx+1] = min(max(_recent_return(5), -0.5), 0.5)
        obs[idx+2] = min(max(_recent_return(20), -0.5), 0.5)

        # Apply nan_to_num and clip inline to the whole array
        np.nan_to_num(obs, nan=0.0, posinf=clip, neginf=-clip, copy=False)
        np.clip(obs, -clip, clip, out=obs)

        return obs

tickers = [f"TICK{i}" for i in range(10)]
n_features = 30
feature_arrays = {t: np.random.randn(1000, n_features).astype(np.float32) for t in tickers}
macro_array = np.random.randn(1000, 5).astype(np.float32)

builder_old = ObservationBuilder(
    tickers=tickers,
    feature_columns=[f"F{i}" for i in range(n_features)],
    feature_arrays=feature_arrays,
    macro_array=macro_array,
    n_macro_features=5
)

builder_fast = FastObservationBuilder(
    tickers=tickers,
    feature_columns=[f"F{i}" for i in range(n_features)],
    feature_arrays=feature_arrays,
    macro_array=macro_array,
    n_macro_features=5
)

portfolio = {t: 10.0 for t in tickers}
prices = {t: 100.0 for t in tickers}
entry_prices = {t: 90.0 for t in tickers}
portfolio_value_history = deque(np.linspace(9000, 10000, 30))

def run_bench(b):
    start = time.time()
    for _ in range(10000):
        b.build(
            current_step=500,
            portfolio=portfolio,
            prices=prices,
            equity=10000.0,
            balance=1000.0,
            initial_balance=10000.0,
            peak_value=11000.0,
            entry_prices=entry_prices,
            portfolio_value_history=portfolio_value_history
        )
    return time.time() - start

# Correctness check
obs_old = builder_old.build(500, portfolio, prices, 10000.0, 1000.0, 10000.0, 11000.0, entry_prices, portfolio_value_history)
obs_fast = builder_fast.build(500, portfolio, prices, 10000.0, 1000.0, 10000.0, 11000.0, entry_prices, portfolio_value_history)
assert np.allclose(obs_old, obs_fast), "Mismatch between old and fast builders"

# Warmup
run_bench(builder_old)
run_bench(builder_fast)

# Measure
t_old = run_bench(builder_old)
t_fast = run_bench(builder_fast)
print(f"Time for 10k calls (Old): {t_old:.4f}s")
print(f"Time for 10k calls (Fast): {t_fast:.4f}s")
print(f"Speedup: {t_old / t_fast:.2f}x")
