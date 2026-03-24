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

        # Pre-allocate contiguous numpy buffer for performance (Zero-allocation build phase)
        self._buffer = np.zeros(self.obs_size, dtype=np.float32)

        # Pre-compute slice indices to avoid repeated overhead
        self._feature_slices = []
        start = 0
        for _ in range(self.n_assets):
            end = start + self.n_features
            self._feature_slices.append(slice(start, end))
            start = end

        self._macro_slice = slice(start, start + self.n_macro_features)
        start += self.n_macro_features

        self._position_slices = []
        for i in range(self.n_assets):
            self._position_slices.append(start + i)
        start += self.n_assets

        self._portfolio_slice = slice(start, start + 3)

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
        buf = self._buffer

        # Technical features per ticker
        for i, ticker in enumerate(self.tickers):
            features_array = self.feature_arrays[ticker]
            if current_step >= len(features_array):
                buf[self._feature_slices[i]] = 0.0
            else:
                buf[self._feature_slices[i]] = features_array[current_step]

        # Macro features (shared across tickers)
        if self.macro_array is not None:
            if current_step < len(self.macro_array):
                buf[self._macro_slice] = self.macro_array[current_step]
            else:
                buf[self._macro_slice] = 0.0

        # Positions
        for i, ticker in enumerate(self.tickers):
            price = prices.get(ticker, 0.0)
            if price > 0:
                position_value = portfolio.get(ticker, 0.0) * price
                position_pct = position_value / (equity + EQUITY_EPSILON)
            else:
                position_pct = 0.0
            buf[self._position_slices[i]] = position_pct

        # Portfolio state
        cash_pct = balance / (equity + EQUITY_EPSILON)
        total_return = (equity - initial_balance) / initial_balance
        drawdown = (peak_value - equity) / (peak_value + EQUITY_EPSILON)

        buf[self._portfolio_slice] = [cash_pct, total_return, drawdown]

        # Apply global nan-to-num and clip once over the entire pre-allocated array (huge speedup)
        np.nan_to_num(buf, nan=0.0, posinf=clip, neginf=-clip, copy=False)
        np.clip(buf, -clip, clip, out=buf)

        # Restore strict bound logic for state elements (as they represent bounded percentages)
        for idx in self._position_slices:
            buf[idx] = max(0.0, min(1.0, buf[idx]))

        ps_start = self._portfolio_slice.start
        buf[ps_start] = max(0.0, min(1.0, buf[ps_start]))          # cash_pct
        buf[ps_start + 1] = max(-1.0, min(5.0, buf[ps_start + 1])) # total_return
        buf[ps_start + 2] = max(0.0, min(1.0, buf[ps_start + 2]))  # drawdown

        return buf.copy()  # Return copy to prevent environment mutating internal buffer
