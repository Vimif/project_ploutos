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
        # Pre-allocate array for speed
        obs = np.empty(self.obs_size, dtype=np.float32)
        idx = 0

        # Technical features per ticker
        for ticker in self.tickers:
            features_array = self.feature_arrays[ticker]
            if current_step >= len(features_array):
                obs[idx:idx+self.n_features] = 0.0
            else:
                obs[idx:idx+self.n_features] = features_array[current_step]
            idx += self.n_features

        # Macro features (shared across tickers)
        if self.macro_array is not None:
            if current_step < len(self.macro_array):
                obs[idx:idx+self.n_macro_features] = self.macro_array[current_step]
            else:
                obs[idx:idx+self.n_macro_features] = 0.0
            idx += self.n_macro_features

        # Position percentages
        for ticker in self.tickers:
            price = prices.get(ticker, 0.0)
            if price > 0:
                position_value = portfolio.get(ticker, 0.0) * price
                val = position_value / (equity + EQUITY_EPSILON)
                obs[idx] = min(max(val, 0.0), 1.0)
            else:
                obs[idx] = 0.0
            idx += 1

        # Unrealized PnL per position
        if entry_prices is None:
            entry_prices = {}
        for ticker in self.tickers:
            entry = entry_prices.get(ticker, 0.0)
            qty = portfolio.get(ticker, 0.0)
            price = prices.get(ticker, 0.0)
            if entry > 0 and qty > 0 and price > 0:
                val = (price - entry) / entry
                obs[idx] = min(max(val, -1.0), 5.0)
            else:
                obs[idx] = 0.0
            idx += 1

        # Portfolio state
        val_cash = balance / (equity + EQUITY_EPSILON)
        obs[idx] = min(max(val_cash, 0.0), 1.0)

        val_ret = (equity - initial_balance) / initial_balance
        obs[idx+1] = min(max(val_ret, -1.0), 5.0)

        val_dd = (peak_value - equity) / (peak_value + EQUITY_EPSILON)
        obs[idx+2] = min(max(val_dd, 0.0), 1.0)
        idx += 3

        # Recent portfolio returns (1-step, 5-step, 20-step)
        if portfolio_value_history is not None:
            hist_len = len(portfolio_value_history)
            def _recent_return(lookback):
                if hist_len > lookback and portfolio_value_history[-lookback - 1] > 0:
                    return (portfolio_value_history[-1] - portfolio_value_history[-lookback - 1]) / portfolio_value_history[-lookback - 1]
                return 0.0
            obs[idx] = min(max(_recent_return(1), -0.5), 0.5)
            obs[idx+1] = min(max(_recent_return(5), -0.5), 0.5)
            obs[idx+2] = min(max(_recent_return(20), -0.5), 0.5)
        else:
            obs[idx:idx+3] = 0.0
        idx += 3

        # Global array replacement for nan to num and clip for speed
        np.nan_to_num(obs, nan=0.0, posinf=clip, neginf=-clip, copy=False)
        np.clip(obs, -clip, clip, out=obs)

        return obs
