"""Tests for AdvancedTransactionModel."""

import numpy as np
import pandas as pd
import pytest

from core.transaction_costs import AdvancedTransactionModel


@pytest.fixture
def model():
    return AdvancedTransactionModel(rng=np.random.RandomState(42))


@pytest.fixture
def recent_prices():
    np.random.seed(42)
    return pd.Series(100 + np.cumsum(np.random.randn(30) * 0.5))


class TestSlippage:
    def test_slippage_with_no_prices_returns_midpoint(self, model):
        slippage = model._calculate_slippage("AAPL", None)
        expected = (model.min_slippage + model.max_slippage) / 2
        assert slippage == expected

    def test_slippage_with_short_series_returns_midpoint(self, model):
        short = pd.Series([100, 101, 102])
        slippage = model._calculate_slippage("AAPL", short)
        expected = (model.min_slippage + model.max_slippage) / 2
        assert slippage == expected

    def test_slippage_increases_with_volatility(self, model):
        # Calm market
        calm = pd.Series(100 + np.arange(30) * 0.01)
        slippage_calm = model._calculate_slippage("AAPL", calm)

        # Volatile market
        np.random.seed(99)
        volatile = pd.Series(100 + np.cumsum(np.random.randn(30) * 5))
        slippage_volatile = model._calculate_slippage("NVDA", volatile)

        assert slippage_volatile > slippage_calm

    def test_slippage_bounded(self, model, recent_prices):
        slippage = model._calculate_slippage("AAPL", recent_prices)
        assert model.min_slippage <= slippage <= model.max_slippage

    def test_vol_ceiling_configurable(self):
        model = AdvancedTransactionModel(vol_ceiling=0.10)
        assert model.vol_ceiling == 0.10


class TestMarketImpact:
    def test_zero_volume_returns_max_slippage(self, model):
        impact = model._calculate_market_impact(100, 0)
        assert impact == model.max_slippage

    def test_negative_volume_returns_max_slippage(self, model):
        impact = model._calculate_market_impact(100, -1)
        assert impact == model.max_slippage

    def test_small_order_low_impact(self, model):
        impact = model._calculate_market_impact(100, 50_000_000)
        assert impact < 0.001

    def test_large_order_higher_impact(self, model):
        small = model._calculate_market_impact(100, 10_000_000)
        large = model._calculate_market_impact(10_000, 10_000_000)
        assert large > small

    def test_impact_clipped(self, model):
        impact = model._calculate_market_impact(1_000_000, 1)
        assert impact <= model.max_slippage


class TestExecutionPrice:
    def test_buy_price_higher_than_intended(self, model, recent_prices):
        exec_price, costs = model.calculate_execution_price(
            "AAPL", 150.0, 100, 5_000_000, "buy", recent_prices
        )
        assert exec_price > 150.0
        assert costs["total_cost"] > 0

    def test_sell_price_lower_than_intended(self, model, recent_prices):
        exec_price, costs = model.calculate_execution_price(
            "AAPL", 150.0, 100, 5_000_000, "sell", recent_prices
        )
        assert exec_price < 150.0

    def test_costs_breakdown_keys(self, model, recent_prices):
        _, costs = model.calculate_execution_price(
            "AAPL", 150.0, 100, 5_000_000, "buy", recent_prices
        )
        assert "commission" in costs
        assert "slippage" in costs
        assert "market_impact" in costs
        assert "latency" in costs
        assert "total_cost" in costs
        assert "total_cost_dollars" in costs

    def test_deterministic_with_seed(self):
        m1 = AdvancedTransactionModel(rng=np.random.RandomState(42))
        m2 = AdvancedTransactionModel(rng=np.random.RandomState(42))
        prices = pd.Series(np.linspace(100, 110, 30))
        p1, _ = m1.calculate_execution_price("X", 100.0, 50, 1_000_000, "buy", prices)
        p2, _ = m2.calculate_execution_price("X", 100.0, 50, 1_000_000, "buy", prices)
        assert p1 == p2
