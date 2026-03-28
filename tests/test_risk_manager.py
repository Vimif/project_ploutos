import pytest
from unittest.mock import MagicMock
from core.risk_manager import RiskManager


@pytest.fixture
def risk_manager():
    return RiskManager()


def test_initialization(risk_manager):
    assert risk_manager is not None


def test_calculate_position_size(risk_manager):
    quantity, value = risk_manager.calculate_position_size(
        portfolio_value=100000.0, entry_price=100.0, stop_loss_pct=0.05
    )
    assert isinstance(quantity, int)
    assert isinstance(value, float)
    assert quantity > 0


def test_check_daily_loss_limit(risk_manager):
    risk_manager.reset_daily_stats(100000)
    assert (
        risk_manager.check_daily_loss_limit(99000) is True
    )  # 1% loss is within 3% limit so it returns True (allowed)
    assert (
        risk_manager.check_daily_loss_limit(90000) is False
    )  # 10% loss is beyond 3% limit so it returns False (circuit breaker)


def test_calculate_portfolio_exposure(risk_manager):
    positions = [{"market_value": 10000}, {"market_value": 20000}]
    exposure = risk_manager.calculate_portfolio_exposure(positions, 100000)
    assert exposure == 0.3


def test_should_reduce_exposure(risk_manager):
    positions = [{"market_value": 90000, "unrealized_plpc": -0.01}]  # 90% exposure > 85%
    should, reason = risk_manager.should_reduce_exposure(positions, 100000)
    assert should is True


def test_calculate_kelly_criterion(risk_manager):
    f = risk_manager.calculate_kelly_criterion(0.6, 200, 100)
    assert isinstance(f, float)


def test_calculate_sharpe_ratio(risk_manager):
    returns = [0.01, 0.02, -0.01, 0.03]
    sharpe = risk_manager.calculate_sharpe_ratio(returns)
    assert isinstance(sharpe, float)


def test_calculate_max_drawdown(risk_manager):
    values = [100, 110, 90, 120, 80, 150]
    mdd, dur = risk_manager.calculate_max_drawdown(values)
    assert isinstance(mdd, float)
    assert mdd < 0


def test_assess_position_risk(risk_manager):
    risk = risk_manager.assess_position_risk("AAPL", 10000, 100000, 0.1, 5)
    assert isinstance(risk, dict)


def test_get_risk_report(risk_manager):
    report = risk_manager.get_risk_report([], 100000)
    assert isinstance(report, dict)
