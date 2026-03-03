import pytest
from core.risk_manager import RiskManager


def test_risk_manager_init():
    rm = RiskManager(
        max_portfolio_risk=0.02, max_daily_loss=0.03, max_position_size=0.05, max_correlation=0.7
    )
    assert rm.max_portfolio_risk == 0.02
    assert rm.max_daily_loss == 0.03
    assert rm.max_position_size == 0.05


def test_check_daily_loss_limit():
    rm = RiskManager(
        max_portfolio_risk=0.02, max_daily_loss=0.03, max_position_size=0.05, max_correlation=0.7
    )
    rm.daily_start_equity = 100000
    assert rm.check_daily_loss_limit(current_value=100000) is True

    # Limit is 3% loss (so <= 97k)
    assert rm.check_daily_loss_limit(current_value=96000) is False


def test_calculate_position_size():
    rm = RiskManager(
        max_portfolio_risk=0.02, max_daily_loss=0.03, max_position_size=1.0, max_correlation=0.7
    )
    shares, dollars = rm.calculate_position_size(
        portfolio_value=100000, entry_price=100, stop_loss_pct=0.05
    )
    assert shares == 400
    assert dollars == 40000


def test_calculate_portfolio_exposure():
    rm = RiskManager()
    positions = [
        {"symbol": "AAPL", "market_value": 10000},
        {"symbol": "MSFT", "market_value": 5000},
    ]
    exposure = rm.calculate_portfolio_exposure(positions, 100000)
    assert exposure == 0.15


def test_calculate_kelly_criterion():
    rm = RiskManager()
    kelly = rm.calculate_kelly_criterion(0.5, 200, 100)
    assert kelly == 0.02


def test_calculate_max_drawdown():
    rm = RiskManager()
    values = [100, 110, 90, 120, 80]
    md_pct, md_dollar = rm.calculate_max_drawdown(values)
    assert round(md_pct, 4) == -0.3333
    assert md_dollar == 40.0


def test_should_reduce_exposure():
    rm = RiskManager(max_daily_loss=0.03)
    rm.daily_start_equity = 100000

    positions = [{"symbol": "AAPL", "market_value": 90000, "unrealized_plpc": -0.01}]
    reduce, reason = rm.should_reduce_exposure(positions, 100000)
    assert reduce is True


def test_assess_position_risk():
    rm = RiskManager(max_position_size=0.1)
    result = rm.assess_position_risk("AAPL", 15000, 100000, unrealized_plpc=-0.01, days_held=1)
    assert result["risk_level"] == "ÉLEVÉ"


def test_risk_manager_extra():
    rm = RiskManager()

    result = rm.assess_position_risk("AAPL", 2000, 100000, 0.05, 1)
    assert result["risk_level"] == "FAIBLE"

    result = rm.assess_position_risk("AAPL", 50000, 100000, -0.05, 10)
    assert result["risk_level"] == "ÉLEVÉ"

    positions = [
        {"symbol": "AAPL", "market_value": 15000, "unrealized_plpc": 0.05, "days_held": 5},
        {"symbol": "MSFT", "market_value": 5000, "unrealized_plpc": -0.02, "days_held": 2},
    ]
    report = rm.get_risk_report(portfolio_value=100000, positions=positions)
    assert report["exposure_pct"] == 0.20
    assert report["positions_count"] == 2


def test_print_risk_summary(caplog):
    rm = RiskManager()
    report = {
        "portfolio_value": 100000,
        "exposure_pct": 0.5,
        "positions_count": 2,
        "risky_positions_count": 0,
        "risky_positions": [],
        "daily_pnl_pct": 0.01,
        "daily_pl": 100,
        "daily_pl_pct": 0.01,
        "daily_trades": 0,
        "daily_wins": 0,
        "daily_losses": 0,
        "circuit_breaker": False,
        "sharpe_ratio": 1.5,
        "max_daily_loss": 0.03,
    }
    rm.print_risk_summary(report)
    assert "RISK MANAGEMENT REPORT" in caplog.text
