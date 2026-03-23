# core/observation_builder.py
"""Observation vector construction for TradingEnv."""

import numpy as np
from typing import Dict, List, Optional

from core.constants import EQUITY_EPSILON, OBSERVATION_CLIP_RANGE


class ObservationBuilder:
    """Builds observation vectors from feature arrays, macro data, and portfolio state."""

    def __init__(
        self,
        tickers: List[str],
        feature_columns: List[str],
        feature_arrays: Dict[str, np.ndarray],
        macro_array: Optional[np.ndarray] = None,
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

        # Pre-allocate buffer and compute slices for performance (Bolt optimization)
        self._obs_buffer = np.zeros(self.obs_size, dtype=np.float32)

        self._ticker_slices = []
        current_idx = 0

        # 1. Technical features slices
        for _ in self.tickers:
            next_idx = current_idx + self.n_features
            self._ticker_slices.append(slice(current_idx, next_idx))
            current_idx = next_idx

        # 2. Macro features slice
        self._macro_slice = slice(current_idx, current_idx + self.n_macro_features)
        current_idx += self.n_macro_features

        # 3. Positions slices start index
        self._pos_start_idx = current_idx
        current_idx += self.n_assets

        # 4. Portfolio state slice
        self._portfolio_slice = slice(current_idx, current_idx + 3)

    def build(
        self,
        current_step: int,
        portfolio: Dict[str, float],
        prices: Dict[str, float],
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

        # 1. Technical features per ticker
        for i, ticker in enumerate(self.tickers):
            features_array = self.feature_arrays[ticker]
            if current_step >= len(features_array):
                # We can just leave it as zeros if we clear it, but arrays
                # might be dirty from previous steps. Safe to overwrite with 0.
                self._obs_buffer[self._ticker_slices[i]] = 0.0
            else:
                self._obs_buffer[self._ticker_slices[i]] = features_array[current_step]

        # 2. Macro features (shared across tickers)
        if self.macro_array is not None:
            if current_step < len(self.macro_array):
                self._obs_buffer[self._macro_slice] = self.macro_array[current_step]
            else:
                self._obs_buffer[self._macro_slice] = 0.0

        # 3. Positions
        for i, ticker in enumerate(self.tickers):
            price = prices.get(ticker, 0.0)
            if price > 0:
                position_value = portfolio.get(ticker, 0.0) * price
                position_pct = position_value / (equity + EQUITY_EPSILON)
            else:
                position_pct = 0.0
            self._obs_buffer[self._pos_start_idx + i] = np.clip(position_pct, 0, 1)

        # 4. Portfolio state
        cash_pct = np.clip(balance / (equity + EQUITY_EPSILON), 0, 1)
        total_return = np.clip((equity - initial_balance) / initial_balance, -1, 5)
        drawdown = np.clip((peak_value - equity) / (peak_value + EQUITY_EPSILON), 0, 1)

        self._obs_buffer[self._portfolio_slice] = [cash_pct, total_return, drawdown]

        # Apply global nan_to_num and clip in-place for performance
        np.nan_to_num(self._obs_buffer, copy=False, nan=0.0, posinf=clip, neginf=-clip)
        np.clip(self._obs_buffer, -clip, clip, out=self._obs_buffer)

        # Return a copy to prevent the environment step from mutating our internal buffer
        return self._obs_buffer.copy()
