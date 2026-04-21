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
        idx = 0

        # Technical features per ticker
        for ticker in self.tickers:
            features_array = self.feature_arrays[ticker]
            if current_step >= len(features_array):
                features = np.zeros(self.n_features, dtype=np.float32)
            else:
                features = features_array[current_step]
            size = self.n_features
            np.clip(
                np.nan_to_num(features, nan=0.0, posinf=clip, neginf=-clip),
                -clip,
                clip,
                out=self.obs_buffer[idx : idx + size],
            )
            idx += size

        # Macro features (shared across tickers)
        if self.macro_array is not None:
            if current_step < len(self.macro_array):
                macro_features = self.macro_array[current_step]
            else:
                macro_features = np.zeros(self.n_macro_features, dtype=np.float32)
            size = self.n_macro_features
            np.clip(
                np.nan_to_num(macro_features, nan=0.0, posinf=clip, neginf=-clip),
                -clip,
                clip,
                out=self.obs_buffer[idx : idx + size],
            )
            idx += size

        # Position percentages
        for ticker in self.tickers:
            price = prices.get(ticker, 0.0)
            if price > 0:
                position_value = portfolio.get(ticker, 0.0) * price
                position_pct = position_value / (equity + EQUITY_EPSILON)
            else:
                position_pct = 0.0
            self.obs_buffer[idx] = min(max(position_pct, 0.0), 1.0)
            idx += 1

        # Unrealized PnL per position
        if entry_prices is None:
            entry_prices = {}
        for ticker in self.tickers:
            entry = entry_prices.get(ticker, 0.0)
            qty = portfolio.get(ticker, 0.0)
            price = prices.get(ticker, 0.0)
            if entry > 0 and qty > 0 and price > 0:
                unrealized_pnl = (price - entry) / entry
            else:
                unrealized_pnl = 0.0
            self.obs_buffer[idx] = min(max(unrealized_pnl, -1.0), 5.0)
            idx += 1

        # Portfolio state
        self.obs_buffer[idx] = min(max(balance / (equity + EQUITY_EPSILON), 0.0), 1.0)
        idx += 1
        self.obs_buffer[idx] = min(max((equity - initial_balance) / initial_balance, -1.0), 5.0)
        idx += 1
        self.obs_buffer[idx] = min(max((peak_value - equity) / (peak_value + EQUITY_EPSILON), 0.0), 1.0)
        idx += 1

        # Recent portfolio returns (1-step, 5-step, 20-step)
        if portfolio_value_history:
            n_hist = len(portfolio_value_history)

            if n_hist > 1 and portfolio_value_history[-2] > 0:
                ret_1 = (portfolio_value_history[-1] - portfolio_value_history[-2]) / portfolio_value_history[-2]
            else:
                ret_1 = 0.0

            if n_hist > 5 and portfolio_value_history[-6] > 0:
                ret_5 = (portfolio_value_history[-1] - portfolio_value_history[-6]) / portfolio_value_history[-6]
            else:
                ret_5 = 0.0

            if n_hist > 20 and portfolio_value_history[-21] > 0:
                ret_20 = (portfolio_value_history[-1] - portfolio_value_history[-21]) / portfolio_value_history[-21]
            else:
                ret_20 = 0.0
        else:
            ret_1, ret_5, ret_20 = 0.0, 0.0, 0.0

        self.obs_buffer[idx] = min(max(ret_1, -0.5), 0.5)
        idx += 1
        self.obs_buffer[idx] = min(max(ret_5, -0.5), 0.5)
        idx += 1
        self.obs_buffer[idx] = min(max(ret_20, -0.5), 0.5)
        idx += 1

        return self.obs_buffer.copy()

