# core/observation_builder.py
"""Observation vector construction for TradingEnv."""

import numpy as np

from core.constants import EQUITY_EPSILON, OBSERVATION_CLIP_RANGE


class ObservationBuilder:
    """Builds observation vectors from feature arrays, macro data, and portfolio state."""

    def __init__(
        self,
        tickers: list[str],
        feature_columns: list[str],
        feature_arrays: dict[str, np.ndarray],
        macro_array: np.ndarray | None = None,
        n_macro_features: int = 0,
    ):
        self.tickers = tickers
        self.n_features = len(feature_columns)
        self.feature_arrays = feature_arrays
        self.macro_array = macro_array
        self.n_macro_features = n_macro_features
        self.n_assets = len(tickers)

        self.obs_size = (
            self.n_assets * self.n_features
            + self.n_macro_features
            + self.n_assets  # positions
            + 3  # cash_pct, total_return, drawdown
        )

        # Pre-allocate contiguous observation buffer for performance
        self._obs_buffer = np.zeros(self.obs_size, dtype=np.float32)

        # Pre-compute slice indices
        self._feature_slices = []
        idx = 0
        for _ in self.tickers:
            self._feature_slices.append(slice(idx, idx + self.n_features))
            idx += self.n_features

        self._macro_slice = slice(idx, idx + self.n_macro_features)
        idx += self.n_macro_features

        self._positions_slice = slice(idx, idx + self.n_assets)
        idx += self.n_assets

        self._portfolio_slice = slice(idx, idx + 3)

    def build(
        self,
        current_step: int,
        portfolio: dict[str, float],
        prices: dict[str, float],
        equity: float,
        balance: float,
        initial_balance: float,
        peak_value: float,
    ) -> np.ndarray:
        """Build observation vector for current step.

        Args:
            current_step: Current timestep index.
            portfolio: Dict of ticker -> quantity held.
            prices: Dict of ticker -> current price.
            equity: Current total portfolio value.
            balance: Current cash balance.
            initial_balance: Starting balance.
            peak_value: Historical peak equity.

        Returns:
            Flat numpy observation vector.
        """
        clip = OBSERVATION_CLIP_RANGE
        obs = self._obs_buffer

        # Technical features per ticker
        for i, ticker in enumerate(self.tickers):
            features_array = self.feature_arrays[ticker]
            if current_step >= len(features_array):
                obs[self._feature_slices[i]] = 0.0
            else:
                obs[self._feature_slices[i]] = features_array[current_step]

        # Macro features (shared across tickers)
        if self.macro_array is not None:
            if current_step < len(self.macro_array):
                obs[self._macro_slice] = self.macro_array[current_step]
            else:
                obs[self._macro_slice] = 0.0

        # Positions
        positions_start = self._positions_slice.start
        for i, ticker in enumerate(self.tickers):
            price = prices.get(ticker, 0.0)
            if price > 0:
                position_value = portfolio.get(ticker, 0.0) * price
                obs[positions_start + i] = position_value / (equity + EQUITY_EPSILON)
            else:
                obs[positions_start + i] = 0.0

        # Portfolio state
        p_idx = self._portfolio_slice.start
        obs[p_idx] = balance / (equity + EQUITY_EPSILON)
        # Note: total_return is clipped to [-1, 5] in the final general clip to OBSERVATION_CLIP_RANGE (10.0),
        # However, the original code clipped it explicitly to [-1, 5] before. To maintain exact behavior, we must replicate this.
        total_ret = (equity - initial_balance) / initial_balance
        obs[p_idx + 1] = max(-1.0, min(5.0, total_ret))
        obs[p_idx + 2] = (peak_value - equity) / (peak_value + EQUITY_EPSILON)

        # Apply transformations once over contiguous memory
        np.nan_to_num(obs, nan=0.0, posinf=clip, neginf=-clip, copy=False)
        np.clip(obs, -clip, clip, out=obs)

        # Position clipping [0,1], cash_pct [0,1], drawdown [0,1]
        # We enforce their specific ranges here to exactly match previous logic
        # Positions
        np.clip(obs[self._positions_slice], 0.0, 1.0, out=obs[self._positions_slice])
        # Cash pct
        obs[p_idx] = max(0.0, min(1.0, obs[p_idx]))
        # Drawdown
        obs[p_idx + 2] = max(0.0, min(1.0, obs[p_idx + 2]))

        # We need to return a copy since buffer is reused in subsequent steps
        return obs.copy()
