from __future__ import annotations

from collections import deque
import importlib
import sys

import numpy as np
import pandas as pd

module = sys.modules.get("core.live_observation")
if module is not None and getattr(module, "__file__", None) is None:
    del sys.modules["core.live_observation"]

LiveObservationEngine = importlib.import_module("core.live_observation").LiveObservationEngine


def _make_feature_frame(index: pd.DatetimeIndex, feature_count: int) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "Open": np.linspace(100, 120, len(index)),
            "High": np.linspace(101, 121, len(index)),
            "Low": np.linspace(99, 119, len(index)),
            "Close": np.linspace(100, 120, len(index)),
            "Volume": np.linspace(1_000_000, 1_100_000, len(index)),
        },
        index=index,
    )
    for idx in range(feature_count):
        base[f"feat_{idx:02d}"] = np.linspace(idx, idx + 1, len(index))
    return base


def test_live_observation_engine_infers_contract_from_model_obs_size(monkeypatch):
    tickers = [f"TICK{i}" for i in range(10)]
    index = pd.date_range("2026-03-01", periods=80, freq="h")
    market_data = {ticker: _make_feature_frame(index, 50) for ticker in tickers}
    macro_data = pd.DataFrame(
        {f"macro_{idx:02d}": np.linspace(idx, idx + 0.5, len(index)) for idx in range(25)},
        index=index,
    )

    engine = LiveObservationEngine(
        tickers=tickers,
        initial_balance=100_000.0,
        max_features_per_ticker=30,
        target_observation_size=468,
    )
    engine.peak_value = 100_000.0
    engine.portfolio_value_history = deque([100_000.0], maxlen=20)
    monkeypatch.setattr(
        LiveObservationEngine,
        "_compute_features",
        lambda self, live_market_data: live_market_data,
    )
    engine.macro_fetcher = type(
        "AlignedMacroFetcher",
        (),
        {
            "align_to_ticker": staticmethod(
                lambda macro_frame, ticker_frame: macro_frame.reindex(ticker_frame.index)
                .ffill()
                .bfill()
            )
        },
    )()

    snapshot = engine.build_snapshot(
        market_data,
        positions_map={},
        balance=100_000.0,
        equity=100_000.0,
        macro_data=macro_data,
    )

    assert snapshot.observation.shape == (468,)
    assert len(snapshot.feature_columns) == 42
    assert len(snapshot.macro_columns) == 22
