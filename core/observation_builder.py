# core/observation_builder.py
"""Observation vector construction for TradingEnv."""

from typing import Dict, List, Optional

import numpy as np

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

        # Performance optimization: pre-allocate a single buffer
        self._obs_buffer = np.zeros(self.obs_size, dtype=np.float32)

        # Precompute slices for fast insertion
        self._feature_slices = []
        idx = 0
        for _ in self.tickers:
            self._feature_slices.append(slice(idx, idx + self.n_features))
            idx += self.n_features

        if self.n_macro_features > 0:
            self._macro_slice = slice(idx, idx + self.n_macro_features)
            idx += self.n_macro_features
        else:
            self._macro_slice = None

        self._pos_start_idx = idx
        self._portfolio_state_start_idx = idx + self.n_assets

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

        # Technical features per ticker
        for i, ticker in enumerate(self.tickers):
            features_array = self.feature_arrays[ticker]
            if current_step >= len(features_array):
                obs[self._feature_slices[i]] = 0.0
            else:
                obs[self._feature_slices[i]] = features_array[current_step]

        # Macro features (shared across tickers)
        if self.macro_array is not None and self._macro_slice is not None:
            if current_step < len(self.macro_array):
                obs[self._macro_slice] = self.macro_array[current_step]
            else:
                obs[self._macro_slice] = 0.0

        # Global nan_to_num and clip on features and macro in-place
        end_features_idx = self._pos_start_idx
        np.nan_to_num(obs[:end_features_idx], nan=0.0, posinf=clip, neginf=-clip, copy=False)
        np.clip(obs[:end_features_idx], -clip, clip, out=obs[:end_features_idx])

        # Positions
        idx = self._pos_start_idx
        denom = equity + EQUITY_EPSILON
        for ticker in self.tickers:
            price = prices.get(ticker, 0.0)
            if price > 0:
                position_value = portfolio.get(ticker, 0.0) * price
                position_pct = position_value / denom
            else:
                position_pct = 0.0
            # Faster scalar clamping using built-ins instead of np.clip
            obs[idx] = max(0.0, min(1.0, position_pct))
            idx += 1

        # Portfolio state
        cash_pct = max(0.0, min(1.0, balance / denom))
        total_return = max(-1.0, min(5.0, (equity - initial_balance) / initial_balance))
        drawdown = max(0.0, min(1.0, (peak_value - equity) / (peak_value + EQUITY_EPSILON)))

        obs[self._portfolio_state_start_idx] = cash_pct
        obs[self._portfolio_state_start_idx + 1] = total_return
        obs[self._portfolio_state_start_idx + 2] = drawdown

        # Return a copy to prevent state aliasing bugs in RL replay buffers
        return obs.copy()
