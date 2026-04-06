# core/observation_builder.py
"""Observation vector construction for TradingEnv."""

import numpy as np
from collections import deque
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
            + self.n_assets  # position percentages
            + self.n_assets  # unrealized PnL per position
            + 3  # cash_pct, total_return, drawdown
            + 3  # recent returns: 1-step, 5-step, 20-step
        )
        self.obs_buffer = np.zeros(self.obs_size, dtype=np.float32)

        # Pre-compute slice indices
        self.feature_slices = []
        idx = 0
        for _ in self.tickers:
            self.feature_slices.append(slice(idx, idx + self.n_features))
            idx += self.n_features

        if self.macro_array is not None:
            self.macro_slice = slice(idx, idx + self.n_macro_features)
            idx += self.n_macro_features
        else:
            self.macro_slice = None

        self.pos_pct_slice = slice(idx, idx + self.n_assets)
        idx += self.n_assets
        self.unrealized_pnl_slice = slice(idx, idx + self.n_assets)
        idx += self.n_assets
        self.portfolio_state_slice = slice(idx, idx + 3)
        idx += 3
        self.recent_returns_slice = slice(idx, idx + 3)

    def build(
        self,
        current_step: int,
        portfolio: Dict[str, float],
        prices: Dict[str, float],
        equity: float,
        balance: float,
        initial_balance: float,
        peak_value: float,
        entry_prices: Optional[Dict[str, float]] = None,
        portfolio_value_history: Optional[deque] = None,
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
            entry_prices: Dict of ticker -> entry price for open positions.
            portfolio_value_history: Recent equity history for return calculation.

        Returns:
            Flat numpy observation vector.
        """
        clip = OBSERVATION_CLIP_RANGE

        # Technical features per ticker
        for i, ticker in enumerate(self.tickers):
            features_array = self.feature_arrays[ticker]
            if current_step >= len(features_array):
                self.obs_buffer[self.feature_slices[i]] = 0.0
            else:
                self.obs_buffer[self.feature_slices[i]] = features_array[current_step]

        # Macro features (shared across tickers)
        if self.macro_slice is not None:
            if current_step < len(self.macro_array):
                self.obs_buffer[self.macro_slice] = self.macro_array[current_step]
            else:
                self.obs_buffer[self.macro_slice] = 0.0

        # Position percentages
        pos_pcts = self.obs_buffer[self.pos_pct_slice]
        for i, ticker in enumerate(self.tickers):
            price = prices.get(ticker, 0.0)
            if price > 0:
                position_pct = (portfolio.get(ticker, 0.0) * price) / (equity + EQUITY_EPSILON)
                pos_pcts[i] = min(1.0, max(0.0, position_pct))
            else:
                pos_pcts[i] = 0.0

        # Unrealized PnL per position
        if entry_prices is None:
            entry_prices = {}
        pnls = self.obs_buffer[self.unrealized_pnl_slice]
        for i, ticker in enumerate(self.tickers):
            entry = entry_prices.get(ticker, 0.0)
            qty = portfolio.get(ticker, 0.0)
            price = prices.get(ticker, 0.0)
            if entry > 0 and qty > 0 and price > 0:
                unrealized_pnl = (price - entry) / entry
                pnls[i] = min(5.0, max(-1.0, unrealized_pnl))
            else:
                pnls[i] = 0.0

        # Portfolio state
        state = self.obs_buffer[self.portfolio_state_slice]
        cash_pct = balance / (equity + EQUITY_EPSILON)
        total_return = (equity - initial_balance) / initial_balance
        drawdown = (peak_value - equity) / (peak_value + EQUITY_EPSILON)
        state[0] = min(1.0, max(0.0, cash_pct))
        state[1] = min(5.0, max(-1.0, total_return))
        state[2] = min(1.0, max(0.0, drawdown))

        # Recent portfolio returns (1-step, 5-step, 20-step)
        ret_slice = self.obs_buffer[self.recent_returns_slice]
        hist = list(portfolio_value_history) if portfolio_value_history else []

        def _recent_return(lookback):
            if len(hist) > lookback and hist[-lookback - 1] > 0:
                return (hist[-1] - hist[-lookback - 1]) / hist[-lookback - 1]
            return 0.0

        ret_slice[0] = min(0.5, max(-0.5, _recent_return(1)))
        ret_slice[1] = min(0.5, max(-0.5, _recent_return(5)))
        ret_slice[2] = min(0.5, max(-0.5, _recent_return(20)))

        # Global NaN and Inf handling for the whole buffer
        np.nan_to_num(self.obs_buffer, nan=0.0, posinf=clip, neginf=-clip, copy=False)
        np.clip(self.obs_buffer, -clip, clip, out=self.obs_buffer)

        return self.obs_buffer.copy()

