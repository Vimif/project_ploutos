import numpy as np
import time
from typing import Dict, List, Optional
from core.constants import OBSERVATION_CLIP_RANGE

# Mock data
tickers = [f"TICKER_{i}" for i in range(10)]
feature_columns = [f"F_{i}" for i in range(50)]
n_steps = 1000

feature_arrays = {t: np.random.randn(n_steps, 50).astype(np.float32) for t in tickers}
macro_array = np.random.randn(n_steps, 10).astype(np.float32)

class ObservationBuilderOld:
    def __init__(self, tickers, feature_columns, feature_arrays, macro_array, n_macro_features):
        self.tickers = tickers
        self.n_features = len(feature_columns)
        self.feature_arrays = feature_arrays
        self.macro_array = macro_array
        self.n_macro_features = n_macro_features
        self.n_assets = len(tickers)

        self.obs_size = (
            self.n_assets * self.n_features
            + self.n_macro_features
            + self.n_assets
            + 3
        )

    def build(self, current_step, portfolio, prices, equity, balance, initial_balance, peak_value):
        clip = 10.0 # OBSERVATION_CLIP_RANGE
        EQUITY_EPSILON = 1e-6
        obs_parts = []

        for ticker in self.tickers:
            features_array = self.feature_arrays[ticker]
            if current_step >= len(features_array):
                features = np.zeros(self.n_features, dtype=np.float32)
            else:
                features = features_array[current_step]
            features = np.nan_to_num(features, nan=0.0, posinf=clip, neginf=-clip)
            features = np.clip(features, -clip, clip)
            obs_parts.append(features)

        if self.macro_array is not None:
            if current_step < len(self.macro_array):
                macro_features = self.macro_array[current_step]
            else:
                macro_features = np.zeros(self.n_macro_features, dtype=np.float32)
            macro_features = np.nan_to_num(macro_features, nan=0.0, posinf=clip, neginf=-clip)
            macro_features = np.clip(macro_features, -clip, clip)
            obs_parts.append(macro_features)

        for ticker in self.tickers:
            price = prices.get(ticker, 0.0)
            if price > 0:
                position_value = portfolio.get(ticker, 0.0) * price
                position_pct = position_value / (equity + EQUITY_EPSILON)
            else:
                position_pct = 0.0
            obs_parts.append([np.clip(position_pct, 0, 1)])

        cash_pct = np.clip(balance / (equity + EQUITY_EPSILON), 0, 1)
        total_return = np.clip((equity - initial_balance) / initial_balance, -1, 5)
        drawdown = np.clip((peak_value - equity) / (peak_value + EQUITY_EPSILON), 0, 1)
        obs_parts.append([cash_pct, total_return, drawdown])

        obs = np.concatenate([np.array(p).flatten() for p in obs_parts])
        obs = np.nan_to_num(obs, nan=0.0, posinf=clip, neginf=-clip)
        obs = np.clip(obs, -clip, clip)

        return obs.astype(np.float32)


class ObservationBuilderNew:
    def __init__(self, tickers, feature_columns, feature_arrays, macro_array, n_macro_features):
        self.tickers = tickers
        self.n_features = len(feature_columns)
        self.feature_arrays = feature_arrays
        self.macro_array = macro_array
        self.n_macro_features = n_macro_features
        self.n_assets = len(tickers)

        self.obs_size = (
            self.n_assets * self.n_features
            + self.n_macro_features
            + self.n_assets
            + 3
        )
        # Pre-allocate obs_buffer and ticker_indices to avoid repeated dictionary/list lookups
        self._obs_buffer = np.zeros(self.obs_size, dtype=np.float32)

        # Pre-compute fixed indices
        self._ticker_indices = []
        idx = 0
        for _ in self.tickers:
            self._ticker_indices.append((idx, idx + self.n_features))
            idx += self.n_features

        self._macro_idx_start = idx
        self._macro_idx_end = idx + self.n_macro_features
        idx += self.n_macro_features

        self._pos_idx_start = idx
        self._pos_idx_end = idx + self.n_assets
        idx += self.n_assets

        self._state_idx_start = idx
        # Pre-cache array views for fast assignment
        # Not needed since we assign by slice/index anyway, but we could avoid some loop overhead

    def build(self, current_step, portfolio, prices, equity, balance, initial_balance, peak_value):
        clip = 10.0 # OBSERVATION_CLIP_RANGE
        EQUITY_EPSILON = 1e-6

        # Technical features
        for (start, end), ticker in zip(self._ticker_indices, self.tickers):
            features_array = self.feature_arrays[ticker]
            if current_step >= len(features_array):
                self._obs_buffer[start:end] = 0.0
            else:
                self._obs_buffer[start:end] = features_array[current_step]

        # Macro features
        if self.macro_array is not None:
            if current_step < len(self.macro_array):
                self._obs_buffer[self._macro_idx_start:self._macro_idx_end] = self.macro_array[current_step]
            else:
                self._obs_buffer[self._macro_idx_start:self._macro_idx_end] = 0.0

        # Positions
        idx = self._pos_idx_start
        for ticker in self.tickers:
            price = prices.get(ticker, 0.0)
            if price > 0:
                position_value = portfolio.get(ticker, 0.0) * price
                self._obs_buffer[idx] = np.clip(position_value / (equity + EQUITY_EPSILON), 0, 1)
            else:
                self._obs_buffer[idx] = 0.0
            idx += 1

        # Portfolio state
        idx = self._state_idx_start
        self._obs_buffer[idx] = np.clip(balance / (equity + EQUITY_EPSILON), 0, 1)
        self._obs_buffer[idx+1] = np.clip((equity - initial_balance) / initial_balance, -1, 5)
        self._obs_buffer[idx+2] = np.clip((peak_value - equity) / (peak_value + EQUITY_EPSILON), 0, 1)

        # In-place nan/inf handling and clipping over the entire buffer at once
        np.nan_to_num(self._obs_buffer, nan=0.0, posinf=clip, neginf=-clip, copy=False)
        np.clip(self._obs_buffer, -clip, clip, out=self._obs_buffer)

        # Return a copy to prevent env modifications from corrupting the buffer
        return self._obs_buffer.copy()

# Setup test variables
portfolio = {t: 10.0 for t in tickers}
prices = {t: 150.0 for t in tickers}
equity = 100000.0
balance = 10000.0
initial_balance = 100000.0
peak_value = 110000.0

builder_old = ObservationBuilderOld(tickers, feature_columns, feature_arrays, macro_array, 10)
builder_new = ObservationBuilderNew(tickers, feature_columns, feature_arrays, macro_array, 10)

# Validate output is same
obs_old = builder_old.build(5, portfolio, prices, equity, balance, initial_balance, peak_value)
obs_new = builder_new.build(5, portfolio, prices, equity, balance, initial_balance, peak_value)

assert np.allclose(obs_old, obs_new), "Outputs differ!"
print("Outputs match!")

# Benchmark Old
t0 = time.time()
for _ in range(10000):
    builder_old.build(5, portfolio, prices, equity, balance, initial_balance, peak_value)
t_old = time.time() - t0

# Benchmark New
t0 = time.time()
for _ in range(10000):
    builder_new.build(5, portfolio, prices, equity, balance, initial_balance, peak_value)
t_new = time.time() - t0

print(f"Old: {t_old:.4f}s")
print(f"New: {t_new:.4f}s")
print(f"Speedup: {t_old / t_new:.2f}x")
