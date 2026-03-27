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

        # Pre-allocate contiguous buffer for high-performance zero-copy updates
        self.obs_buffer = np.zeros(self.obs_size, dtype=np.float32)

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

        idx = 0
        # Technical features per ticker
        for ticker in self.tickers:
            features_array = self.feature_arrays[ticker]
            if current_step >= len(features_array):
                # We can just leave it as zeros which was pre-allocated, but to be safe and clear:
                self.obs_buffer[idx : idx + self.n_features] = 0.0
            else:
                self.obs_buffer[idx : idx + self.n_features] = features_array[current_step]
            idx += self.n_features

        # Macro features (shared across tickers)
        if self.macro_array is not None:
            if current_step < len(self.macro_array):
                self.obs_buffer[idx : idx + self.n_macro_features] = self.macro_array[current_step]
            else:
                self.obs_buffer[idx : idx + self.n_macro_features] = 0.0
            idx += self.n_macro_features

        # Positions
        for ticker in self.tickers:
            price = prices.get(ticker, 0.0)
            if price > 0:
                position_value = portfolio.get(ticker, 0.0) * price
                position_pct = position_value / (equity + EQUITY_EPSILON)
            else:
                position_pct = 0.0

            # Fast scalar clip [0, 1] using built-in min/max
            position_pct_clipped = max(0.0, min(position_pct, 1.0))

            self.obs_buffer[idx] = position_pct_clipped
            idx += 1

        # Portfolio state
        cash_pct = balance / (equity + EQUITY_EPSILON)
        cash_pct_clipped = max(0.0, min(cash_pct, 1.0))

        total_return = (equity - initial_balance) / initial_balance
        total_return_clipped = max(-1.0, min(total_return, 5.0))

        drawdown = (peak_value - equity) / (peak_value + EQUITY_EPSILON)
        drawdown_clipped = max(0.0, min(drawdown, 1.0))

        self.obs_buffer[idx] = cash_pct_clipped
        self.obs_buffer[idx + 1] = total_return_clipped
        self.obs_buffer[idx + 2] = drawdown_clipped

        # Global NaN/Inf replacement and clipping in-place
        np.nan_to_num(self.obs_buffer, nan=0.0, posinf=clip, neginf=-clip, copy=False)
        np.clip(self.obs_buffer, -clip, clip, out=self.obs_buffer)

        # Return a copy to prevent state aliasing bugs in RL replay buffers
        return self.obs_buffer.copy()
