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

        # Pre-allocate buffer and slice indices for ultra-fast build()
        self._obs_buffer = np.zeros(self.obs_size, dtype=np.float32)

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
        for (start, end), ticker in zip(self._ticker_indices, self.tickers, strict=True):
            features_array = self.feature_arrays[ticker]
            if current_step >= len(features_array):
                self._obs_buffer[start:end] = 0.0
            else:
                self._obs_buffer[start:end] = features_array[current_step]

        # Macro features (shared across tickers)
        if self.macro_array is not None:
            if current_step < len(self.macro_array):
                self._obs_buffer[self._macro_idx_start : self._macro_idx_end] = self.macro_array[
                    current_step
                ]
            else:
                self._obs_buffer[self._macro_idx_start : self._macro_idx_end] = 0.0

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
        self._obs_buffer[idx + 1] = np.clip((equity - initial_balance) / initial_balance, -1, 5)
        self._obs_buffer[idx + 2] = np.clip(
            (peak_value - equity) / (peak_value + EQUITY_EPSILON), 0, 1
        )

        # In-place nan/inf handling and clipping over the entire buffer at once
        np.nan_to_num(self._obs_buffer, nan=0.0, posinf=clip, neginf=-clip, copy=False)
        np.clip(self._obs_buffer, -clip, clip, out=self._obs_buffer)

        # Return a copy to prevent env modifications from corrupting the buffer
        return self._obs_buffer.copy()
