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

        # Pre-allocate the entire observation array
        self.obs_array = np.zeros(self.obs_size, dtype=np.float32)

        # Pre-calculate slice indices to avoid computation in the loop
        self.slice_indices = []
        idx = 0
        for _ in self.tickers:
            self.slice_indices.append((idx, idx + self.n_features))
            idx += self.n_features

        self.macro_start_idx = idx
        self.macro_end_idx = idx + self.n_macro_features
        idx += self.n_macro_features

        self.pos_start_idx = idx

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
        obs = self.obs_array

        # Technical features per ticker
        for i, ticker in enumerate(self.tickers):
            features_array = self.feature_arrays[ticker]
            start, end = self.slice_indices[i]
            if current_step >= len(features_array):
                obs[start:end] = 0.0
            else:
                obs[start:end] = features_array[current_step]

        # Macro features (shared across tickers)
        if self.macro_array is not None:
            if current_step < len(self.macro_array):
                obs[self.macro_start_idx : self.macro_end_idx] = self.macro_array[current_step]
            else:
                obs[self.macro_start_idx : self.macro_end_idx] = 0.0

        # Positions
        idx = self.pos_start_idx
        equity_eps = equity + EQUITY_EPSILON

        for ticker in self.tickers:
            price = prices.get(ticker, 0.0)
            if price > 0:
                position_value = portfolio.get(ticker, 0.0) * price
                position_pct = position_value / equity_eps
            else:
                position_pct = 0.0

            obs[idx] = max(0.0, min(1.0, position_pct))
            idx += 1

        # Portfolio state
        obs[idx] = max(0.0, min(1.0, balance / equity_eps))
        idx += 1
        obs[idx] = max(-1.0, min(5.0, (equity - initial_balance) / initial_balance))
        idx += 1
        obs[idx] = max(0.0, min(1.0, (peak_value - equity) / (peak_value + EQUITY_EPSILON)))

        # Apply nan_to_num and clip over the entire contiguous memory block exactly once
        obs = np.nan_to_num(obs, nan=0.0, posinf=clip, neginf=-clip)
        np.clip(obs, -clip, clip, out=obs)

        return obs
