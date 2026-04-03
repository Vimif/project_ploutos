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

        # Pre-allocate buffer
        self.obs_buffer = np.zeros(self.obs_size, dtype=np.float32)

        # Pre-compute slices for fast access
        self.slices = {}
        idx = 0
        for ticker in self.tickers:
            self.slices[ticker] = slice(idx, idx + self.n_features)
            idx += self.n_features

        self.macro_slice = slice(idx, idx + self.n_macro_features)
        idx += self.n_macro_features

        self.positions_start = idx
        idx += self.n_assets
        self.portfolio_start = idx

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

        # Technical features per ticker
        for ticker in self.tickers:
            features_array = self.feature_arrays[ticker]
            if current_step >= len(features_array):
                self.obs_buffer[self.slices[ticker]] = 0.0
            else:
                self.obs_buffer[self.slices[ticker]] = features_array[current_step]

        # Macro features (shared across tickers)
        if self.macro_array is not None:
            if current_step < len(self.macro_array):
                self.obs_buffer[self.macro_slice] = self.macro_array[current_step]
            else:
                self.obs_buffer[self.macro_slice] = 0.0

        # Global NaN and Clip for features in-place
        features_end = self.positions_start
        np.nan_to_num(self.obs_buffer[:features_end], nan=0.0, posinf=clip, neginf=-clip, copy=False)
        np.clip(self.obs_buffer[:features_end], -clip, clip, out=self.obs_buffer[:features_end])

        # Positions
        idx = self.positions_start
        for ticker in self.tickers:
            price = prices.get(ticker, 0.0)
            if price > 0:
                position_value = portfolio.get(ticker, 0.0) * price
                position_pct = position_value / (equity + EQUITY_EPSILON)
            else:
                position_pct = 0.0
            self.obs_buffer[idx] = max(0.0, min(1.0, position_pct))
            idx += 1

        # Portfolio state
        cash_pct = balance / (equity + EQUITY_EPSILON)
        total_return = (equity - initial_balance) / initial_balance
        drawdown = (peak_value - equity) / (peak_value + EQUITY_EPSILON)

        idx = self.portfolio_start
        self.obs_buffer[idx] = max(0.0, min(1.0, cash_pct))
        self.obs_buffer[idx+1] = max(-1.0, min(5.0, total_return))
        self.obs_buffer[idx+2] = max(0.0, min(1.0, drawdown))

        return self.obs_buffer.copy()
