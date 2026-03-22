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

        # Pre-allocate buffer and compute slices for performance
        self._obs_buffer = np.zeros(self.obs_size, dtype=np.float32)

        self._ticker_slices = []
        start_idx = 0
        for _ in tickers:
            end_idx = start_idx + self.n_features
            self._ticker_slices.append(slice(start_idx, end_idx))
            start_idx = end_idx

        if self.n_macro_features > 0:
            self._macro_slice = slice(start_idx, start_idx + self.n_macro_features)
            start_idx += self.n_macro_features
        else:
            self._macro_slice = None

        self._positions_slice = slice(start_idx, start_idx + self.n_assets)
        start_idx += self.n_assets

        self._portfolio_slice = slice(start_idx, start_idx + 3)

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
        clip = float(OBSERVATION_CLIP_RANGE)

        # Technical features per ticker
        for i, ticker in enumerate(self.tickers):
            features_array = self.feature_arrays[ticker]
            if current_step >= len(features_array):
                self._obs_buffer[self._ticker_slices[i]] = 0.0
            else:
                self._obs_buffer[self._ticker_slices[i]] = features_array[current_step]

        # Macro features (shared across tickers)
        if self._macro_slice is not None:
            if current_step < len(self.macro_array):
                self._obs_buffer[self._macro_slice] = self.macro_array[current_step]
            else:
                self._obs_buffer[self._macro_slice] = 0.0

        # Positions
        pos_idx = self._positions_slice.start
        equity_denom = equity + EQUITY_EPSILON
        for ticker in self.tickers:
            price = prices.get(ticker, 0.0)
            if price > 0:
                self._obs_buffer[pos_idx] = (portfolio.get(ticker, 0.0) * price) / equity_denom
            else:
                self._obs_buffer[pos_idx] = 0.0
            pos_idx += 1

        # Portfolio state
        port_idx = self._portfolio_slice.start
        self._obs_buffer[port_idx] = balance / equity_denom
        self._obs_buffer[port_idx + 1] = (equity - initial_balance) / initial_balance
        self._obs_buffer[port_idx + 2] = (peak_value - equity) / (peak_value + EQUITY_EPSILON)

        # Global nan_to_num and clip in-place for performance
        np.nan_to_num(self._obs_buffer, copy=False, nan=0.0, posinf=clip, neginf=-clip)
        np.clip(self._obs_buffer, -clip, clip, out=self._obs_buffer)

        # Apply specific constraints for positions and portfolio metrics
        np.clip(
            self._obs_buffer[self._positions_slice],
            0.0,
            1.0,
            out=self._obs_buffer[self._positions_slice],
        )
        self._obs_buffer[port_idx] = np.clip(self._obs_buffer[port_idx], 0.0, 1.0)
        self._obs_buffer[port_idx + 1] = np.clip(self._obs_buffer[port_idx + 1], -1.0, 5.0)
        self._obs_buffer[port_idx + 2] = np.clip(self._obs_buffer[port_idx + 2], 0.0, 1.0)

        return self._obs_buffer.copy()
