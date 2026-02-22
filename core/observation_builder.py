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
        obs_parts = []

        # Technical features per ticker
        for ticker in self.tickers:
            features_array = self.feature_arrays[ticker]
            if current_step >= len(features_array):
                features = np.zeros(self.n_features, dtype=np.float32)
            else:
                features = features_array[current_step]
            features = np.nan_to_num(features, nan=0.0, posinf=clip, neginf=-clip)
            features = np.clip(features, -clip, clip)
            obs_parts.append(features)

        # Macro features (shared across tickers)
        if self.macro_array is not None:
            if current_step < len(self.macro_array):
                macro_features = self.macro_array[current_step]
            else:
                macro_features = np.zeros(self.n_macro_features, dtype=np.float32)
            macro_features = np.nan_to_num(macro_features, nan=0.0, posinf=clip, neginf=-clip)
            macro_features = np.clip(macro_features, -clip, clip)
            obs_parts.append(macro_features)

        # Positions
        for ticker in self.tickers:
            price = prices.get(ticker, 0.0)
            if price > 0:
                position_value = portfolio.get(ticker, 0.0) * price
                position_pct = position_value / (equity + EQUITY_EPSILON)
            else:
                position_pct = 0.0
            obs_parts.append([np.clip(position_pct, 0, 1)])

        # Portfolio state
        cash_pct = np.clip(balance / (equity + EQUITY_EPSILON), 0, 1)
        total_return = np.clip((equity - initial_balance) / initial_balance, -1, 5)
        drawdown = np.clip((peak_value - equity) / (peak_value + EQUITY_EPSILON), 0, 1)
        obs_parts.append([cash_pct, total_return, drawdown])

        obs = np.concatenate([np.array(p).flatten() for p in obs_parts])
        obs = np.nan_to_num(obs, nan=0.0, posinf=clip, neginf=-clip)
        obs = np.clip(obs, -clip, clip)

        return obs.astype(np.float32)
