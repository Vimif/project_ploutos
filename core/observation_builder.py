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

        # ⚡ Bolt Optimization: Pre-allocate observation array and slice indices
        # to avoid repeated object creation and np.concatenate() in the hot loop.
        self._obs = np.zeros(self.obs_size, dtype=np.float32)

        self._feat_slices = []
        idx = 0
        for _ in self.tickers:
            self._feat_slices.append(slice(idx, idx + self.n_features))
            idx += self.n_features

        self._macro_slice = slice(idx, idx + self.n_macro_features)
        idx += self.n_macro_features

        self._pos_slice = slice(idx, idx + self.n_assets)
        idx += self.n_assets

        self._state_slice = slice(idx, idx + 3)

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
        for i, ticker in enumerate(self.tickers):
            features_array = self.feature_arrays[ticker]
            if current_step >= len(features_array):
                self._obs[self._feat_slices[i]] = 0.0
            else:
                self._obs[self._feat_slices[i]] = features_array[current_step]

        # Macro features (shared across tickers)
        if self.macro_array is not None:
            if current_step < len(self.macro_array):
                self._obs[self._macro_slice] = self.macro_array[current_step]
            else:
                self._obs[self._macro_slice] = 0.0

        # Positions
        pos_idx = self._pos_slice.start
        for ticker in self.tickers:
            price = prices.get(ticker, 0.0)
            if price > 0:
                position_value = portfolio.get(ticker, 0.0) * price
                self._obs[pos_idx] = position_value / (equity + EQUITY_EPSILON)
            else:
                self._obs[pos_idx] = 0.0
            pos_idx += 1

        # Portfolio state
        self._obs[self._state_slice.start] = balance / (equity + EQUITY_EPSILON)
        self._obs[self._state_slice.start + 1] = (equity - initial_balance) / initial_balance
        self._obs[self._state_slice.start + 2] = (peak_value - equity) / (
            peak_value + EQUITY_EPSILON
        )

        # ⚡ Bolt Optimization: Batch NaN handling and clipping in-place
        np.nan_to_num(self._obs, nan=0.0, posinf=clip, neginf=-clip, copy=False)
        np.clip(self._obs, -clip, clip, out=self._obs)

        # Enforce specific bounds for positions and state variables
        np.clip(self._obs[self._pos_slice], 0, 1, out=self._obs[self._pos_slice])
        self._obs[self._state_slice.start] = np.clip(self._obs[self._state_slice.start], 0, 1)
        self._obs[self._state_slice.start + 1] = np.clip(
            self._obs[self._state_slice.start + 1], -1, 5
        )
        self._obs[self._state_slice.start + 2] = np.clip(
            self._obs[self._state_slice.start + 2], 0, 1
        )

        return self._obs.copy()
