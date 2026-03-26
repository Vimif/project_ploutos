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

        # Performance optimization: pre-allocate memory and slice indices
        self._obs_buffer = np.zeros(self.obs_size, dtype=np.float32)

        self._feat_slices = []
        start = 0
        for _ in tickers:
            end = start + self.n_features
            self._feat_slices.append(slice(start, end))
            start = end

        self._macro_slice = slice(start, start + self.n_macro_features)
        start += self.n_macro_features

        self._pos_slice = slice(start, start + self.n_assets)
        start += self.n_assets

        self._state_slice = slice(start, start + 3)

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
        obs = self._obs_buffer

        # Technical features per ticker
        for i, ticker in enumerate(self.tickers):
            features_array = self.feature_arrays[ticker]
            if current_step >= len(features_array):
                obs[self._feat_slices[i]] = 0.0
            else:
                obs[self._feat_slices[i]] = features_array[current_step]

        # Macro features (shared across tickers)
        if self.macro_array is not None:
            if current_step < len(self.macro_array):
                obs[self._macro_slice] = self.macro_array[current_step]
            else:
                obs[self._macro_slice] = 0.0

        # Positions
        pos_idx = self._pos_slice.start
        for ticker in self.tickers:
            price = prices.get(ticker, 0.0)
            if price > 0:
                position_value = portfolio.get(ticker, 0.0) * price
                position_pct = position_value / (equity + EQUITY_EPSILON)
            else:
                position_pct = 0.0

            # Fast scalar clip, fallback to np.clip for arrays
            if isinstance(position_pct, (int, float, np.number)):
                if position_pct < 0.0:
                    position_pct = 0.0
                elif position_pct > 1.0:
                    position_pct = 1.0
            else:
                position_pct = np.clip(position_pct, 0.0, 1.0)

            obs[pos_idx] = position_pct
            pos_idx += 1

        # Portfolio state
        cash_pct = balance / (equity + EQUITY_EPSILON)
        if isinstance(cash_pct, (int, float, np.number)):
            if cash_pct < 0.0:
                cash_pct = 0.0
            elif cash_pct > 1.0:
                cash_pct = 1.0
        else:
            cash_pct = np.clip(cash_pct, 0.0, 1.0)

        total_return = (equity - initial_balance) / initial_balance
        if isinstance(total_return, (int, float, np.number)):
            if total_return < -1.0:
                total_return = -1.0
            elif total_return > 5.0:
                total_return = 5.0
        else:
            total_return = np.clip(total_return, -1.0, 5.0)

        drawdown = (peak_value - equity) / (peak_value + EQUITY_EPSILON)
        if isinstance(drawdown, (int, float, np.number)):
            if drawdown < 0.0:
                drawdown = 0.0
            elif drawdown > 1.0:
                drawdown = 1.0
        else:
            drawdown = np.clip(drawdown, 0.0, 1.0)

        obs[self._state_slice.start] = cash_pct
        obs[self._state_slice.start + 1] = total_return
        obs[self._state_slice.start + 2] = drawdown

        # Apply global operations exactly once at the end over contiguous block
        np.nan_to_num(obs, nan=0.0, posinf=clip, neginf=-clip, copy=False)
        np.clip(obs, -clip, clip, out=obs)

        return obs.copy()
