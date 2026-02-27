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

        # Optimization: Pre-compute list of feature arrays for faster iteration
        # This avoids dictionary lookups inside the hot loop
        self._ticker_feature_arrays = [feature_arrays[t] for t in tickers]

        # Pre-compute slice indices
        self._feature_end = self.n_assets * self.n_features
        self._macro_start = self._feature_end
        self._macro_end = self._macro_start + self.n_macro_features
        self._pos_start = self._macro_end
        self._pos_end = self._pos_start + self.n_assets
        self._global_start = self._pos_end

        self.obs_clip = OBSERVATION_CLIP_RANGE

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
        # Optimization: Pre-allocate single array instead of concatenating lists
        obs = np.zeros(self.obs_size, dtype=np.float32)
        clip = self.obs_clip

        # 1. Technical features (Bulk Copy)
        # Iterate over pre-computed list instead of dict lookups
        feat_idx = 0
        n_feat = self.n_features

        for arr in self._ticker_feature_arrays:
            if current_step < len(arr):
                # Direct slice assignment is much faster than np.concatenate
                obs[feat_idx : feat_idx + n_feat] = arr[current_step]
            feat_idx += n_feat

        # 2. Macro features
        if self.macro_array is not None:
            if current_step < len(self.macro_array):
                obs[self._macro_start : self._macro_end] = self.macro_array[current_step]

        # 3. Positions
        # Vectorized calculation over list comprehension is faster than loop with append
        equity_inv = 1.0 / (equity + EQUITY_EPSILON)

        # Map dict values to array indices corresponding to tickers list order
        # We trust self.tickers order matches observation slots
        pos_vals = [
            (portfolio.get(t, 0.0) * prices.get(t, 0.0) * equity_inv)
            if prices.get(t, 0.0) > 0 else 0.0
            for t in self.tickers
        ]

        # Fix: Explicitly clip position percentages to [0, 1] as in original implementation
        # The global clip later might be looser (e.g. [-10, 10]), so this local clip is important logic.
        obs[self._pos_start : self._pos_end] = np.clip(pos_vals, 0, 1)

        # 4. Portfolio state
        # Clip individual values here as they are scalars
        obs[self._global_start] = np.clip(balance * equity_inv, 0, 1)
        obs[self._global_start + 1] = np.clip((equity - initial_balance) / initial_balance, -1, 5)
        obs[self._global_start + 2] = np.clip((peak_value - equity) / (peak_value + EQUITY_EPSILON), 0, 1)

        # Final pass: Global Clip/NaN handling
        # Doing this once on the full array is significantly faster than on small chunks
        # Using in-place operations where possible
        obs = np.nan_to_num(obs, nan=0.0, posinf=clip, neginf=-clip)
        np.clip(obs, -clip, clip, out=obs)

        return obs
