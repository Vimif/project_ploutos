import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import datetime

from core.risk_manager import RiskManager


@pytest.fixture
def risk_manager():
    return RiskManager(max_portfolio_risk=0.02, max_daily_loss=0.03, max_position_size=0.05)


def test_initialization(risk_manager):
    assert risk_manager.max_portfolio_risk == 0.02
    assert risk_manager.max_daily_loss == 0.03
    assert risk_manager.max_position_size == 0.05
    assert risk_manager.daily_start_value is None


def test_calculate_position_size(risk_manager):
    # Test position sizing logic
    # risk per trade = 2% (default)
    # If account = 100k, risk = 2k
    # If entry = 100, stop = 90 (10% stop)
    # stop distance = 10
    # shares = 2000 / 10 = 200 shares

    entry_price = 100.0
    stop_loss_pct = 0.10
    portfolio_value = 100000.0

    qty, val = risk_manager.calculate_position_size(
        portfolio_value=portfolio_value, entry_price=entry_price, stop_loss_pct=stop_loss_pct
    )

    # Expected: 200 shares based on risk, BUT capped at 5% of 100k = 5000.
    # 5000 / 100 = 50 shares.
    assert qty == 50
    assert val == 5000.0


def test_check_daily_loss_limit(risk_manager):
    # Initial call sets start value
    assert risk_manager.check_daily_loss_limit(100000.0) is True
    assert risk_manager.daily_start_value == 100000.0

    # Small loss (1%) - Allowed
    assert risk_manager.check_daily_loss_limit(99000.0) is True

    # Big loss (4%) - Not Allowed (limit is 3%)
    assert risk_manager.check_daily_loss_limit(96000.0) is False
    assert risk_manager.circuit_breaker_triggered is True


def test_reset_daily_stats(risk_manager):
    risk_manager.daily_trades = 10
    risk_manager.reset_daily_stats(105000.0)
    assert risk_manager.daily_trades == 0
    assert risk_manager.daily_start_value == 105000.0


def test_should_reduce_exposure(risk_manager):
    # Mock portfolio positions
    positions = [
        {"symbol": "AAPL", "market_value": 50000.0, "unrealized_plpc": 0.1},
        {"symbol": "MSFT", "market_value": 40000.0, "unrealized_plpc": 0.05},
    ]
    # Total exposure = 90k / 100k = 90% > 85% limit
    should_reduce, reason = risk_manager.should_reduce_exposure(positions, 100000.0)
    assert should_reduce is True
    assert "Exposition élevée" in reason


def test_kelly_criterion(risk_manager):
    # Win rate 50%, Win/Loss = 2.0 (Win 2%, Loss 1%)
    # Kelly = 0.5 - (0.5 / 2.0) = 0.5 - 0.25 = 0.25 (25%)
    # Half Kelly = 12.5%
    # Capped at max_risk (2%)

    kelly = risk_manager.calculate_kelly_criterion(0.5, 0.02, 0.01)
    assert kelly == 0.02  # Capped


def test_calculate_sharpe_ratio(risk_manager):
    returns = [0.01, -0.005, 0.01, 0.002]  # Very short
    sharpe = risk_manager.calculate_sharpe_ratio(returns)
    assert isinstance(sharpe, float)


def test_max_drawdown(risk_manager):
    values = [100, 110, 100, 90, 100, 120]
    # Peak 110 -> 90. DD = 20 / 110 = 18.18%
    max_dd_pct, max_dd_amt = risk_manager.calculate_max_drawdown(values)
    assert pytest.approx(max_dd_amt) == 20.0
    assert pytest.approx(max_dd_pct) == -0.18181818


def test_assess_position_risk(risk_manager):
    risk = risk_manager.assess_position_risk(
        symbol="TEST",
        position_value=1000,
        portfolio_value=100000,
        unrealized_plpc=-0.15,  # Big loss
        days_held=10,
    )
    assert risk["risk_level"] in ["ÉLEVÉ", "CRITIQUE"]
    assert "Perte importante" in risk["warnings"][0]
