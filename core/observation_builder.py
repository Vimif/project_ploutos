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

        # Pre-allocate contiguous NumPy buffer for zero-copy slice updates
        self._obs_buffer = np.zeros(self.obs_size, dtype=np.float32)

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
        obs = self._obs_buffer
        idx = 0

        # Technical features per ticker
        for ticker in self.tickers:
            features_array = self.feature_arrays[ticker]
            if current_step >= len(features_array):
                obs[idx : idx + self.n_features] = 0.0
            else:
                obs[idx : idx + self.n_features] = features_array[current_step]
            idx += self.n_features

        # Macro features (shared across tickers)
        if self.macro_array is not None:
            if current_step < len(self.macro_array):
                obs[idx : idx + self.n_macro_features] = self.macro_array[current_step]
            else:
                obs[idx : idx + self.n_macro_features] = 0.0
        idx += self.n_macro_features

        # Positions
        equity_plus_eps = equity + EQUITY_EPSILON
        for ticker in self.tickers:
            price = prices.get(ticker, 0.0)
            if price > 0:
                position_value = portfolio.get(ticker, 0.0) * price
                position_pct = position_value / equity_plus_eps
            else:
                position_pct = 0.0

            # Explicit scalar clamping is faster than np.clip
            if position_pct > 1.0:
                position_pct = 1.0
            elif position_pct < 0.0:
                position_pct = 0.0

            obs[idx] = position_pct
            idx += 1

        # Portfolio state
        cash_pct = balance / equity_plus_eps
        if cash_pct > 1.0:
            cash_pct = 1.0
        elif cash_pct < 0.0:
            cash_pct = 0.0

        total_return = (equity - initial_balance) / initial_balance
        if total_return > 5.0:
            total_return = 5.0
        elif total_return < -1.0:
            total_return = -1.0

        drawdown = (peak_value - equity) / (peak_value + EQUITY_EPSILON)
        if drawdown > 1.0:
            drawdown = 1.0
        elif drawdown < 0.0:
            drawdown = 0.0

        obs[idx] = cash_pct
        obs[idx + 1] = total_return
        obs[idx + 2] = drawdown

        # Apply global nan/inf and clipping efficiently in-place
        np.nan_to_num(obs, nan=0.0, posinf=clip, neginf=-clip, copy=False)
        np.clip(obs, -clip, clip, out=obs)

        # Return a copy to prevent replay buffer aliasing
        return obs.copy()
