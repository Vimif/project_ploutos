from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from scripts import performance_benchmark


class FakeLiveObservationEngine:
    def __init__(self, tickers, initial_balance, max_features_per_ticker=0):
        self.tickers = tickers
        self.initial_balance = initial_balance
        self.max_features_per_ticker = max_features_per_ticker

    def _compute_features(self, market_data):
        processed = {}
        for ticker, df in market_data.items():
            enriched = df.copy()
            enriched["feat_close"] = enriched["Close"].pct_change().fillna(0.0)
            rolling_volume = enriched["Volume"].rolling(5).mean().fillna(enriched["Volume"])
            enriched["feat_volume"] = enriched["Volume"] / rolling_volume
            processed[ticker] = enriched
        return processed

    def build_snapshot(self, market_data, positions_map, *, balance, equity, macro_data=None):
        del positions_map, balance, equity
        prices = {ticker: float(df["Close"].iloc[-1]) for ticker, df in market_data.items()}
        volumes = {ticker: float(df["Volume"].iloc[-1]) for ticker, df in market_data.items()}
        recent_prices = {ticker: df["Close"].tail(20) for ticker, df in market_data.items()}
        feature_columns = ["feat_close", "feat_volume"]
        macro_columns = list(macro_data.columns) if macro_data is not None else []
        observation_size = len(self.tickers) * len(feature_columns) + len(macro_columns) + 6
        observation = np.linspace(0.0, 1.0, observation_size)
        return SimpleNamespace(
            observation=observation.astype(np.float32),
            prices=prices,
            volumes=volumes,
            recent_prices=recent_prices,
            feature_columns=feature_columns,
            macro_columns=macro_columns,
        )


class FakeLiveExecutionConfig:
    def __init__(self):
        self.ensemble_size = 3
        self.interval_minutes = 60
        self.buy_pct = 0.15
        self.order_fill_timeout_seconds = 30
        self.order_poll_interval_seconds = 1.0

    @classmethod
    def from_dict(cls, data=None):
        del data
        return cls()


class FakeSimulatedBroker:
    def __init__(self, initial_balance):
        self.cash = float(initial_balance)
        self.positions = {}

    def get_account(self):
        return {
            "cash": self.cash,
            "portfolio_value": self.cash,
            "equity": self.cash,
        }

    def get_positions(self):
        return list(self.positions.values())

    def buy(self, symbol, price, notional):
        qty = float(notional) / float(price)
        self.cash -= float(notional)
        self.positions[symbol] = {
            "symbol": symbol,
            "qty": qty,
            "avg_entry_price": price,
            "current_price": price,
            "market_value": qty * price,
            "cost_basis": qty * price,
            "unrealized_pl": 0.0,
            "unrealized_plpc": 0.0,
        }
        return qty


class FakeLiveBrokerAdapter:
    def __init__(self, broker, broker_name, fill_timeout=30, poll_interval=1.0):
        del broker_name, fill_timeout, poll_interval
        self.broker = broker

    def get_account(self):
        return self.broker.get_account()

    def get_positions_map(self):
        return {position["symbol"]: position for position in self.broker.get_positions()}

    def get_open_orders(self):
        return []

    def estimate_trade_cost(self, symbol, *, notional, price_hint, current_volume, recent_prices):
        del symbol, notional, price_hint, current_volume, recent_prices
        return {"breakdown": {"total_cost": 0.001}}

    def buy_notional(self, symbol, notional, *, price_hint, reason=""):
        del reason
        qty = self.broker.buy(symbol, price_hint, notional)
        return SimpleNamespace(
            success=True,
            symbol=symbol,
            side="buy",
            qty=qty,
            requested_notional=notional,
            filled_price=price_hint,
            status="filled",
            reason="",
        )


class FakeLiveExecutionEngine:
    def __init__(self, broker_adapter, config):
        self.broker = broker_adapter
        self.config = config

    def evaluate_market_regime(self, market_data, macro_data):
        del market_data, macro_data
        return SimpleNamespace(
            risk_on=True,
            reason="risk_on",
            fast_ma=1.0,
            slow_ma=0.9,
            vix=20.0,
        )

    def execute_buy(
        self,
        symbol,
        *,
        confidence,
        price,
        current_volume,
        recent_prices,
        positions_map,
        equity,
        regime,
        reason,
    ):
        del confidence, current_volume, recent_prices, positions_map, regime, reason
        notional = equity * self.config.buy_pct
        result = self.broker.buy_notional(symbol, notional, price_hint=price)
        return result, {"notional": notional}


def make_settings(tmp_path):
    return performance_benchmark.BenchmarkSettings(
        config_path=None,
        output_path=str(tmp_path / "benchmark.json"),
        model_path=None,
        tickers=["AAPL", "SPY", "QQQ"],
        periods=120,
        iterations=2,
        warmup_iterations=0,
        initial_balance=100_000.0,
        max_features_per_ticker=2,
        interval="1h",
        interval_minutes=60,
        ensemble_size=3,
        include_macro=True,
        random_seed=7,
    )


def install_fakes(monkeypatch):
    monkeypatch.setattr(
        performance_benchmark,
        "_import_project_components",
        lambda: {
            "LiveObservationEngine": FakeLiveObservationEngine,
            "LiveBrokerAdapter": FakeLiveBrokerAdapter,
            "LiveExecutionConfig": FakeLiveExecutionConfig,
            "LiveExecutionEngine": FakeLiveExecutionEngine,
            "SimulatedBroker": FakeSimulatedBroker,
        },
    )
    monkeypatch.setattr(
        performance_benchmark,
        "_detect_hardware",
        lambda: {"cpu_count": 8, "ram_gb": 32.0},
    )


def test_run_benchmark_returns_stage_report(monkeypatch, tmp_path):
    install_fakes(monkeypatch)

    settings = make_settings(tmp_path)
    report = performance_benchmark.run_benchmark(settings, config={})

    assert report["predictor_mode"] == "synthetic"
    assert report["observation_size"] > 0
    assert set(report["stages"]) == {
        "feature_compute",
        "snapshot_build",
        "inference",
        "regime_filter",
        "buy_path",
        "end_to_end_cycle",
    }
    assert report["market_snapshot"]["n_tickers"] == 3
    assert report["recommendations"]
    assert len(report["sample_outputs"]["actions"]) == 3


def test_save_report_writes_json(monkeypatch, tmp_path):
    install_fakes(monkeypatch)

    settings = make_settings(tmp_path)
    report = performance_benchmark.run_benchmark(settings, config={})
    output_path = performance_benchmark.save_report(report, settings.output_path)

    with open(output_path, "r", encoding="utf-8") as handle:
        saved = json.load(handle)

    assert output_path.exists()
    assert saved["settings"]["tickers"] == ["AAPL", "SPY", "QQQ"]
