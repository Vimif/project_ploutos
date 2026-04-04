"""Tests for the normalized live execution layer."""

from __future__ import annotations

import sys
import types

import pandas as pd

dotenv_module = types.ModuleType("dotenv")
dotenv_module.load_dotenv = lambda *args, **kwargs: None
sys.modules.setdefault("dotenv", dotenv_module)

from trading.live_execution import (
    LiveBrokerAdapter,
    LiveExecutionConfig,
    LiveExecutionEngine,
    MarketRegime,
    SimulatedBroker,
)


def make_engine():
    broker = SimulatedBroker(initial_balance=100_000)
    adapter = LiveBrokerAdapter(broker, broker_name="simulate")
    config = LiveExecutionConfig()
    engine = LiveExecutionEngine(adapter, config)
    return engine, adapter


def test_should_buy_blocks_low_confidence(monkeypatch):
    engine, adapter = make_engine()
    monkeypatch.setattr(adapter, "get_open_orders", lambda: [])
    monkeypatch.setattr(
        adapter,
        "estimate_trade_cost",
        lambda symbol, **kwargs: {"breakdown": {"total_cost": 0.001}},
    )

    allowed, reason, _ = engine.should_buy(
        "AAPL",
        confidence=0.50,
        price=100.0,
        current_volume=1_000_000,
        recent_prices=pd.Series([99, 100, 101]),
        positions_map={},
        equity=100_000,
        regime=MarketRegime(True, "risk_on", 101.0, 100.0, 18.0),
    )

    assert not allowed
    assert reason == "low_confidence"


def test_should_buy_blocks_when_regime_is_risk_off(monkeypatch):
    engine, adapter = make_engine()
    monkeypatch.setattr(adapter, "get_open_orders", lambda: [])
    monkeypatch.setattr(
        adapter,
        "estimate_trade_cost",
        lambda symbol, **kwargs: {"breakdown": {"total_cost": 0.001}},
    )

    allowed, reason, details = engine.should_buy(
        "AAPL",
        confidence=0.9,
        price=100.0,
        current_volume=1_000_000,
        recent_prices=pd.Series([99, 100, 101]),
        positions_map={},
        equity=100_000,
        regime=MarketRegime(False, "risk_filter_blocked", 95.0, 100.0, 35.0),
    )

    assert not allowed
    assert reason == "risk_filter_blocked"
    assert details["vix"] == 35.0


def test_should_buy_blocks_when_cost_too_high(monkeypatch):
    engine, adapter = make_engine()
    monkeypatch.setattr(adapter, "get_open_orders", lambda: [])
    monkeypatch.setattr(
        adapter,
        "estimate_trade_cost",
        lambda symbol, **kwargs: {"breakdown": {"total_cost": 0.01}},
    )

    allowed, reason, _ = engine.should_buy(
        "AAPL",
        confidence=0.9,
        price=100.0,
        current_volume=1_000_000,
        recent_prices=pd.Series([99, 100, 101]),
        positions_map={},
        equity=100_000,
        regime=MarketRegime(True, "risk_on", 101.0, 100.0, 18.0),
    )

    assert not allowed
    assert reason == "cost_too_high"


def test_should_buy_blocks_when_position_cap_is_reached(monkeypatch):
    engine, adapter = make_engine()
    monkeypatch.setattr(adapter, "get_open_orders", lambda: [])
    monkeypatch.setattr(
        adapter,
        "estimate_trade_cost",
        lambda symbol, **kwargs: {"breakdown": {"total_cost": 0.001}},
    )
    positions_map = {
        f"TICK{i}": {"qty": 1.0, "market_value": 1_000.0}
        for i in range(engine.config.max_open_positions)
    }

    allowed, reason, _ = engine.should_buy(
        "AAPL",
        confidence=0.9,
        price=100.0,
        current_volume=1_000_000,
        recent_prices=pd.Series([99, 100, 101]),
        positions_map=positions_map,
        equity=100_000,
        regime=MarketRegime(True, "risk_on", 101.0, 100.0, 18.0),
    )

    assert not allowed
    assert reason == "max_open_positions"


def test_duplicate_buy_signal_is_blocked(monkeypatch):
    engine, adapter = make_engine()
    adapter.update_market_prices({"AAPL": 100.0})
    monkeypatch.setattr(adapter, "get_open_orders", lambda: [])
    monkeypatch.setattr(
        adapter,
        "estimate_trade_cost",
        lambda symbol, **kwargs: {"breakdown": {"total_cost": 0.001}},
    )

    first, _ = engine.execute_buy(
        "AAPL",
        confidence=0.9,
        price=100.0,
        current_volume=1_000_000,
        recent_prices=pd.Series([99, 100, 101]),
        positions_map={},
        equity=100_000,
        regime=MarketRegime(True, "risk_on", 101.0, 100.0, 18.0),
        reason="unit-test",
    )
    second, _ = engine.execute_buy(
        "AAPL",
        confidence=0.9,
        price=100.0,
        current_volume=1_000_000,
        recent_prices=pd.Series([99, 100, 101]),
        positions_map=adapter.get_positions_map(),
        equity=100_000,
        regime=MarketRegime(True, "risk_on", 101.0, 100.0, 18.0),
        reason="unit-test",
    )

    assert first.success
    assert not second.success
    assert second.reason == "duplicate_signal"


def test_risk_exit_signals_cover_stop_loss_and_take_profit():
    engine, _ = make_engine()
    signals = engine.get_risk_exit_signals(
        {
            "AAPL": {"qty": 1.0, "unrealized_plpc": -0.08},
            "MSFT": {"qty": 1.0, "unrealized_plpc": 0.18},
            "NVDA": {"qty": 1.0, "unrealized_plpc": 0.01},
        }
    )

    assert signals["AAPL"] == "stop_loss"
    assert signals["MSFT"] == "take_profit"
    assert "NVDA" not in signals


def test_market_regime_uses_spy_trend_and_vix():
    engine, _ = make_engine()
    spy = pd.DataFrame({"Close": list(range(1, 80))})
    macro = pd.DataFrame({"vix_close": [22.0, 19.0]})

    regime = engine.evaluate_market_regime({"SPY": spy}, macro)

    assert regime.risk_on
    assert regime.vix == 19.0
