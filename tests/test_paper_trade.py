"""Lightweight tests for the eToro-first paper trading flow."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

for mod in [
    "stable_baselines3",
    "stable_baselines3.common",
    "stable_baselines3.common.vec_env",
    "sb3_contrib",
]:
    sys.modules.setdefault(mod, MagicMock())

dotenv_module = types.ModuleType("dotenv")
dotenv_module.load_dotenv = lambda *args, **kwargs: None
sys.modules.setdefault("dotenv", dotenv_module)

live_observation_module = types.ModuleType("core.live_observation")
live_observation_module.LiveObservationEngine = MagicMock()
sys.modules.setdefault("core.live_observation", live_observation_module)

from scripts import paper_trade
from trading.live_execution import LiveBrokerAdapter, SimulatedBroker


class FakePredictor:
    def predict_with_asset_confidences(self, observation, deterministic=True):
        return np.array([1, 0]), np.array([0.9, 0.95])


class FakeSellPredictor:
    def predict_with_asset_confidences(self, observation, deterministic=True):
        return np.array([2, 2]), np.array([0.9, 0.9])


class FakeObservationEngine:
    def __init__(
        self,
        tickers,
        initial_balance,
        max_features_per_ticker=0,
        target_observation_size=None,
        feature_columns=None,
        macro_columns=None,
    ):
        self.tickers = tickers
        self.target_observation_size = target_observation_size
        self.feature_columns = feature_columns or []
        self.macro_columns = macro_columns or []

    def build_snapshot(self, market_data, positions_map, *, balance, equity, macro_data=None):
        prices = {ticker: float(df["Close"].iloc[-1]) for ticker, df in market_data.items()}
        return SimpleNamespace(
            observation=np.array([1.0, 2.0], dtype=np.float32),
            prices=prices,
            volumes={ticker: float(df["Volume"].iloc[-1]) for ticker, df in market_data.items()},
            recent_prices={ticker: df["Close"].tail(20) for ticker, df in market_data.items()},
            current_step=len(next(iter(market_data.values()))) - 1,
            feature_columns=[],
            macro_columns=list(macro_data.columns) if macro_data is not None else [],
            processed_data=market_data,
            aligned_macro=macro_data,
        )


def make_market_data():
    index = pd.date_range("2026-04-01", periods=80, freq="h")
    spy = pd.DataFrame(
        {
            "Open": np.linspace(100, 120, len(index)),
            "High": np.linspace(101, 121, len(index)),
            "Low": np.linspace(99, 119, len(index)),
            "Close": np.linspace(100, 120, len(index)),
            "Volume": np.full(len(index), 1_000_000),
        },
        index=index,
    )
    aapl = pd.DataFrame(
        {
            "Open": np.linspace(180, 190, len(index)),
            "High": np.linspace(181, 191, len(index)),
            "Low": np.linspace(179, 189, len(index)),
            "Close": np.linspace(180, 190, len(index)),
            "Volume": np.full(len(index), 1_500_000),
        },
        index=index,
    )
    return {"AAPL": aapl, "SPY": spy}


def test_get_model_actions_returns_actions_and_confidence():
    predictor = FakePredictor()

    actions, confidences = paper_trade.get_model_actions(
        predictor,
        np.array([1.0, 2.0], dtype=np.float32),
    )

    np.testing.assert_array_equal(actions, np.array([1, 0]))
    np.testing.assert_array_equal(confidences, np.array([0.9, 0.95]))


def test_run_paper_trading_supports_etoro_mode(monkeypatch, tmp_path):
    market_data = make_market_data()
    macro_data = pd.DataFrame({"vix_close": [22.0, 18.0]})
    adapter = LiveBrokerAdapter(SimulatedBroker(100_000), broker_name="etoro")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        paper_trade,
        "load_runtime_config",
        lambda path: {
            "data": {"interval": "1h", "tickers": ["AAPL", "SPY"]},
            "environment": {"initial_balance": 100_000, "max_features_per_ticker": 1},
            "live": {
                "interval_minutes": 0,
                "ensemble_size": 1,
                "regime_fast_ma": 5,
                "regime_slow_ma": 10,
                "dedupe_window_seconds": 3_300,
            },
        },
    )
    monkeypatch.setattr(
        paper_trade,
        "load_predictor_bundle",
        lambda model_path, live_settings, config, requested_models=None: (
            FakePredictor(),
            ["AAPL", "SPY"],
            2,
            {"data": {"interval": "1h"}, "environment": {"max_features_per_ticker": 1}},
            None,
            [Path(model_path)],
        ),
    )
    monkeypatch.setattr(paper_trade, "fetch_live_data", lambda *args, **kwargs: market_data)
    monkeypatch.setattr(
        paper_trade,
        "fetch_live_macro_data",
        lambda *args, **kwargs: macro_data,
    )
    monkeypatch.setattr(paper_trade, "LiveObservationEngine", FakeObservationEngine)
    monkeypatch.setattr(
        paper_trade,
        "create_live_broker_adapter",
        lambda mode, initial_balance, fill_timeout, poll_interval: adapter,
    )
    monkeypatch.setattr(paper_trade.time, "sleep", lambda seconds: None)

    report = paper_trade.run_paper_trading(
        model_path="model.zip",
        mode="etoro",
        config_path="config.yaml",
        max_hours=1,
        max_iterations=1,
    )

    assert report["mode"] == "etoro"
    assert report["strategy_family"] == "ppo_ensemble"
    assert report["summary"]["n_trades"] == 1
    assert report["summary"]["n_rejections"] == 0
    assert report["resolved_models"] == ["model.zip"]
    session_dir = Path(report["session_dir"])
    assert (session_dir / "session_meta.json").exists()
    assert (session_dir / "events.jsonl").exists()
    assert (session_dir / "equity.jsonl").exists()
    assert (session_dir / "report.json").exists()


def test_live_trade_journal_writes_append_only_session_files(tmp_path):
    session_dir = tmp_path / "logs" / "paper_trading" / "session_001"
    journal = paper_trade.LiveTradeJournal(
        session_dir,
        {
            "mode": "etoro",
            "broker": "etoro",
            "tickers": ["AAPL", "SPY"],
            "live_settings": {"interval_minutes": 60},
        },
    )

    journal.record_signal("AAPL", "BUY", confidence=0.91, price=180.0, reason="model_buy")
    journal.record_trade("AAPL", "BUY", 180.0, 5.0, 900.0, reason="model_buy", confidence=0.91)
    journal.record_rejection("SPY", "BUY", "low_confidence", {"confidence": 0.4})
    journal.record_alert("warning", "drawdown_near_limit", {"drawdown": 0.09})
    journal.record_equity(
        100_500.0,
        balance=99_100.0,
        n_positions=1,
        drawdown=0.01,
        exposure=0.014,
        source="broker",
    )
    summary = journal.get_summary(100_000.0)
    journal.finalize({"session_id": journal.session_id, "summary": summary})

    meta = json.loads((session_dir / "session_meta.json").read_text(encoding="utf-8"))
    events = (session_dir / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    equity = (session_dir / "equity.jsonl").read_text(encoding="utf-8").strip().splitlines()
    report = json.loads((session_dir / "report.json").read_text(encoding="utf-8"))
    legacy_journal = json.loads((session_dir / "journal.json").read_text(encoding="utf-8"))

    assert meta["mode"] == "etoro"
    assert len(events) == 4
    assert len(equity) == 1
    assert report["summary"]["n_trades"] == 1
    assert legacy_journal["trades"][0]["ticker"] == "AAPL"


def test_resolve_vecnormalize_path_falls_back_to_fold_artifact(tmp_path):
    fold_dir = tmp_path / "fold_00"
    fold_dir.mkdir(parents=True)
    model_path = fold_dir / "model.zip"
    model_path.write_text("placeholder", encoding="utf-8")
    vecnorm_path = fold_dir / "vecnormalize.pkl"
    vecnorm_path.write_text("stats", encoding="utf-8")

    resolved = paper_trade.resolve_vecnormalize_path(model_path, None)

    assert resolved == vecnorm_path


def test_run_paper_trading_supports_rule_strategy_without_model(monkeypatch, tmp_path):
    market_data = make_market_data()
    macro_data = pd.DataFrame({"vix_close": [22.0] * len(next(iter(market_data.values())))})
    adapter = LiveBrokerAdapter(SimulatedBroker(100_000), broker_name="simulate")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        paper_trade,
        "load_runtime_config",
        lambda path: {
            "data": {"interval": "4h", "tickers": ["AAPL", "SPY"]},
            "environment": {"initial_balance": 100_000, "max_features_per_ticker": 1},
            "live": {
                "interval_minutes": 0,
                "buy_pct": 0.10,
                "max_position_pct": 0.10,
                "max_open_positions": 4,
                "regime_fast_ma": 5,
                "regime_slow_ma": 10,
                "dedupe_window_seconds": 13_800,
            },
            "strategy": {
                "family": "rule_momentum_regime",
                "rule_fast_ma": 5,
                "rule_slow_ma": 20,
                "rule_momentum_lookback": 3,
            },
        },
    )
    monkeypatch.setattr(
        paper_trade,
        "load_predictor_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("PPO loader should not be used")),
    )
    monkeypatch.setattr(paper_trade, "fetch_live_data", lambda *args, **kwargs: market_data)
    monkeypatch.setattr(paper_trade, "fetch_live_macro_data", lambda *args, **kwargs: macro_data)
    monkeypatch.setattr(paper_trade, "LiveObservationEngine", FakeObservationEngine)
    monkeypatch.setattr(
        paper_trade,
        "create_live_broker_adapter",
        lambda mode, initial_balance, fill_timeout, poll_interval: adapter,
    )
    monkeypatch.setattr(paper_trade.time, "sleep", lambda seconds: None)

    report = paper_trade.run_paper_trading(
        model_path=None,
        mode="simulate",
        config_path="config.yaml",
        max_hours=1,
        max_iterations=1,
    )

    assert report["mode"] == "simulate"
    assert report["strategy_family"] == "rule_momentum_regime"
    assert report["summary"]["n_trades"] == 1
    assert report["resolved_models"] == []


def test_run_paper_trading_skips_sell_signals_without_position(monkeypatch, tmp_path):
    market_data = make_market_data()
    macro_data = pd.DataFrame({"vix_close": [22.0, 18.0]})
    adapter = LiveBrokerAdapter(SimulatedBroker(100_000), broker_name="etoro")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        paper_trade,
        "load_runtime_config",
        lambda path: {
            "data": {"interval": "1h", "tickers": ["AAPL", "SPY"]},
            "environment": {"initial_balance": 100_000, "max_features_per_ticker": 1},
            "live": {
                "interval_minutes": 0,
                "ensemble_size": 1,
                "regime_fast_ma": 5,
                "regime_slow_ma": 10,
                "dedupe_window_seconds": 3_300,
            },
        },
    )
    monkeypatch.setattr(
        paper_trade,
        "load_predictor_bundle",
        lambda model_path, live_settings, config, requested_models=None: (
            FakeSellPredictor(),
            ["AAPL", "SPY"],
            2,
            {"data": {"interval": "1h"}, "environment": {"max_features_per_ticker": 1}},
            None,
            [Path(model_path)],
        ),
    )
    monkeypatch.setattr(paper_trade, "fetch_live_data", lambda *args, **kwargs: market_data)
    monkeypatch.setattr(
        paper_trade,
        "fetch_live_macro_data",
        lambda *args, **kwargs: macro_data,
    )
    monkeypatch.setattr(paper_trade, "LiveObservationEngine", FakeObservationEngine)
    monkeypatch.setattr(
        paper_trade,
        "create_live_broker_adapter",
        lambda mode, initial_balance, fill_timeout, poll_interval: adapter,
    )
    monkeypatch.setattr(paper_trade.time, "sleep", lambda seconds: None)

    report = paper_trade.run_paper_trading(
        model_path="model.zip",
        mode="etoro",
        config_path="config.yaml",
        max_hours=1,
        max_iterations=1,
    )

    assert report["summary"]["n_trades"] == 0
    assert report["summary"]["n_rejections"] == 0
