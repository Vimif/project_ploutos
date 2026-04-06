"""Lightweight observation pipeline for live inference."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from core.constants import PORTFOLIO_HISTORY_WINDOW
from core.features import FeatureEngineer
from core.macro_data import MacroDataFetcher
from core.observation_builder import ObservationBuilder

logger = logging.getLogger(__name__)


@dataclass
class LiveMarketSnapshot:
    """Prepared market snapshot for live inference and execution filters."""

    observation: np.ndarray
    prices: Dict[str, float]
    volumes: Dict[str, float]
    recent_prices: Dict[str, pd.Series]
    current_step: int
    feature_columns: list[str]
    macro_columns: list[str]
    processed_data: Dict[str, pd.DataFrame]
    aligned_macro: Optional[pd.DataFrame]


class LiveObservationEngine:
    """Builds a single observation without recreating a full TradingEnv."""

    def __init__(
        self,
        tickers: list[str],
        initial_balance: float,
        max_features_per_ticker: int = 0,
        target_observation_size: Optional[int] = None,
        feature_columns: Optional[list[str]] = None,
        macro_columns: Optional[list[str]] = None,
    ):
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.initial_balance = float(initial_balance)
        self.max_features_per_ticker = int(max_features_per_ticker)
        self.target_observation_size = (
            int(target_observation_size) if target_observation_size is not None else None
        )
        self.preferred_feature_columns = list(feature_columns or [])
        self.preferred_macro_columns = list(macro_columns or [])
        self.feature_engineer = FeatureEngineer()
        self.macro_fetcher = MacroDataFetcher()
        self.peak_value = float(initial_balance)
        self.portfolio_value_history: deque[float] = deque(maxlen=PORTFOLIO_HISTORY_WINDOW)

    def reset(self) -> None:
        """Reset live rolling state."""
        self.peak_value = float(self.initial_balance)
        self.portfolio_value_history.clear()

    def build_snapshot(
        self,
        market_data: Dict[str, pd.DataFrame],
        positions_map: Dict[str, dict],
        *,
        balance: float,
        equity: float,
        macro_data: Optional[pd.DataFrame] = None,
    ) -> LiveMarketSnapshot:
        """Prepare features and build one live observation."""

        processed_data = self._compute_features(market_data)
        ref_df = processed_data[self.tickers[0]]
        aligned_macro = None
        if macro_data is not None and not macro_data.empty:
            aligned_macro = self.macro_fetcher.align_to_ticker(macro_data, ref_df)

        feature_columns, macro_columns = self._resolve_observation_columns(ref_df, aligned_macro)
        feature_arrays = {
            ticker: np.nan_to_num(
                pd.to_numeric(
                    processed_data[ticker][feature_columns].values.flatten(),
                    errors="coerce",
                ).reshape(len(processed_data[ticker]), len(feature_columns)),
                nan=0.0,
            ).astype(np.float32)
            for ticker in self.tickers
        }

        macro_array = None
        if aligned_macro is not None and not aligned_macro.empty and macro_columns:
            macro_array = aligned_macro[macro_columns].values.astype(np.float32)

        current_step = min(len(processed_data[ticker]) for ticker in self.tickers) - 1
        if macro_array is not None:
            current_step = min(current_step, len(macro_array) - 1)
        if current_step < 0:
            raise ValueError("Not enough market data to build a live observation")

        self.peak_value = max(self.peak_value, float(equity))
        self.portfolio_value_history.append(float(equity))

        prices = {
            ticker: float(processed_data[ticker]["Close"].iloc[current_step])
            for ticker in self.tickers
        }
        volumes = {
            ticker: (
                float(processed_data[ticker]["Volume"].iloc[current_step])
                if "Volume" in processed_data[ticker].columns
                else 1_000_000.0
            )
            for ticker in self.tickers
        }
        recent_prices = {
            ticker: processed_data[ticker]["Close"].iloc[
                max(0, current_step - 20) : current_step + 1
            ]
            for ticker in self.tickers
        }
        portfolio = {
            ticker: float(positions_map.get(ticker, {}).get("qty", 0.0)) for ticker in self.tickers
        }
        entry_prices = {
            ticker: float(positions_map.get(ticker, {}).get("avg_entry_price", 0.0))
            for ticker in self.tickers
        }

        builder = ObservationBuilder(
            tickers=self.tickers,
            feature_columns=feature_columns,
            feature_arrays=feature_arrays,
            macro_array=macro_array,
            n_macro_features=len(macro_columns),
        )
        observation = builder.build(
            current_step=current_step,
            portfolio=portfolio,
            prices=prices,
            equity=float(equity),
            balance=float(balance),
            initial_balance=float(self.initial_balance),
            peak_value=float(self.peak_value),
            entry_prices=entry_prices,
            portfolio_value_history=self.portfolio_value_history,
        )

        return LiveMarketSnapshot(
            observation=observation,
            prices=prices,
            volumes=volumes,
            recent_prices=recent_prices,
            current_step=current_step,
            feature_columns=feature_columns,
            macro_columns=macro_columns,
            processed_data=processed_data,
            aligned_macro=aligned_macro,
        )

    def _compute_features(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        processed = {}
        for ticker in self.tickers:
            if ticker not in market_data:
                raise KeyError(f"Missing live market data for {ticker}")
            processed[ticker] = self.feature_engineer.calculate_all_features(
                market_data[ticker].copy()
            )
        return processed

    def _all_feature_columns(self, ref_df: pd.DataFrame) -> list[str]:
        exclude_cols = {"Open", "High", "Low", "Close", "Volume", "Date", "Datetime", "Timestamp"}
        return [
            col
            for col in ref_df.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(ref_df[col])
        ]

    def _select_feature_columns(
        self,
        ref_df: pd.DataFrame,
        available_feature_columns: Optional[list[str]] = None,
        feature_count: Optional[int] = None,
    ) -> list[str]:
        available = list(available_feature_columns or self._all_feature_columns(ref_df))
        if self.preferred_feature_columns:
            selected = [col for col in self.preferred_feature_columns if col in available]
            if selected:
                return selected

        target_count = feature_count
        if target_count is None and self.max_features_per_ticker > 0:
            target_count = min(self.max_features_per_ticker, len(available))

        if target_count is not None and target_count < len(available):
            variances = ref_df[available].var().fillna(0)
            return variances.nlargest(target_count).index.tolist()
        return available

    def _select_macro_columns(
        self,
        available_macro_columns: list[str],
        macro_count: Optional[int] = None,
    ) -> list[str]:
        available = list(available_macro_columns)
        if not available:
            return []
        if self.preferred_macro_columns:
            selected = [col for col in self.preferred_macro_columns if col in available]
            if selected:
                return selected
        if macro_count is not None:
            return available[:macro_count]
        return available

    def _resolve_observation_columns(
        self,
        ref_df: pd.DataFrame,
        aligned_macro: Optional[pd.DataFrame],
    ) -> tuple[list[str], list[str]]:
        available_feature_columns = self._all_feature_columns(ref_df)
        available_macro_columns = (
            list(aligned_macro.columns)
            if aligned_macro is not None and not aligned_macro.empty
            else []
        )

        feature_columns = self._select_feature_columns(ref_df, available_feature_columns)
        macro_columns = self._select_macro_columns(available_macro_columns)

        if self.target_observation_size is None:
            return feature_columns, macro_columns

        current_size = self._observation_size_for(len(feature_columns), len(macro_columns))
        if current_size == self.target_observation_size:
            return feature_columns, macro_columns

        inferred_feature_columns, inferred_macro_columns = self._infer_columns_for_target_size(
            ref_df,
            available_feature_columns,
            available_macro_columns,
            preferred_feature_count=len(feature_columns),
        )
        inferred_size = self._observation_size_for(
            len(inferred_feature_columns), len(inferred_macro_columns)
        )
        if inferred_size == self.target_observation_size:
            logger.info(
                "Adjusted live observation contract to match model: %s features/ticker + %s macro = %s dims",
                len(inferred_feature_columns),
                len(inferred_macro_columns),
                inferred_size,
            )
            return inferred_feature_columns, inferred_macro_columns

        logger.warning(
            "Unable to infer live observation contract for target %s dims; using %s features/ticker + %s macro (%s dims)",
            self.target_observation_size,
            len(feature_columns),
            len(macro_columns),
            current_size,
        )
        return feature_columns, macro_columns

    def _infer_columns_for_target_size(
        self,
        ref_df: pd.DataFrame,
        available_feature_columns: list[str],
        available_macro_columns: list[str],
        *,
        preferred_feature_count: int,
    ) -> tuple[list[str], list[str]]:
        target_variable_dims = self.target_observation_size - self._static_observation_dims
        if target_variable_dims < 0:
            return self._select_feature_columns(ref_df, available_feature_columns), []

        candidates: list[tuple[int, int]] = []
        max_macro = min(len(available_macro_columns), target_variable_dims)
        for macro_count in range(max_macro, -1, -1):
            remaining = target_variable_dims - macro_count
            if remaining < 0 or remaining % self.n_assets != 0:
                continue
            feature_count = remaining // self.n_assets
            if 0 <= feature_count <= len(available_feature_columns):
                candidates.append((feature_count, macro_count))

        if not candidates:
            return self._select_feature_columns(
                ref_df, available_feature_columns
            ), self._select_macro_columns(available_macro_columns)

        preferred = (
            preferred_feature_count
            if preferred_feature_count > 0
            else min(len(available_feature_columns), max(target_variable_dims // self.n_assets, 0))
        )
        feature_count, macro_count = min(
            candidates,
            key=lambda item: (
                abs(item[0] - preferred),
                -item[1],
                item[0],
            ),
        )
        return (
            self._select_feature_columns(
                ref_df,
                available_feature_columns,
                feature_count=feature_count,
            ),
            self._select_macro_columns(available_macro_columns, macro_count=macro_count),
        )

    @property
    def _static_observation_dims(self) -> int:
        return len(self.tickers) * 2 + 6

    def _observation_size_for(self, feature_count: int, macro_count: int) -> int:
        return len(self.tickers) * feature_count + macro_count + self._static_observation_dims
