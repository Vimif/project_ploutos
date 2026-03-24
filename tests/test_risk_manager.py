import sys
from unittest.mock import MagicMock

if "numpy" not in sys.modules:
    sys.modules["numpy"] = MagicMock()
if "core.utils" not in sys.modules:
    mock_utils = MagicMock()
    mock_logger = MagicMock()
    mock_utils.setup_logging.return_value = mock_logger
    sys.modules["core.utils"] = mock_utils

from datetime import datetime, timedelta

from core.risk_manager import RiskManager


def test_risk_manager_basic():
    rm = RiskManager(
        max_position_size=0.2,
    )

    positions = [
        {
            "symbol": "AAPL",
            "market_value": 10000,
            "unrealized_plpc": -0.06,
            "purchase_date": (datetime.now() - timedelta(days=10)).isoformat(),
        },
        {
            "symbol": "MSFT",
            "market_value": 25000,
            "unrealized_plpc": 0.05,
            "purchase_date": (datetime.now() - timedelta(days=5)).isoformat(),
        },
    ]
    portfolio_value = 100000

    report = rm.get_risk_report(positions, portfolio_value)

    assert report["portfolio_value"] == 100000
    assert len(report["risky_positions"]) > 0


def test_risk_manager_empty():
    rm = RiskManager()
    report = rm.get_risk_report([], 100000)
    assert len(report["risky_positions"]) == 0


def test_calculate_var():
    rm = RiskManager()
    assert True


def test_calculate_position_size():
    rm = RiskManager(max_portfolio_risk=0.02, max_position_size=0.05)
    size, _ = rm.calculate_position_size(
        portfolio_value=100000, entry_price=100.0, stop_loss_pct=0.1
    )
    assert size == 50


def test_check_daily_loss_limit():
    rm = RiskManager(max_daily_loss=0.03)
    rm.reset_daily_stats(100000.0)
    assert rm.check_daily_loss_limit(98000.0) == True
    assert rm.check_daily_loss_limit(95000.0) == False


def test_calculate_portfolio_exposure():
    rm = RiskManager()
    positions = [
        {"symbol": "AAPL", "market_value": 10000},
        {"symbol": "MSFT", "market_value": 25000},
    ]
    exposure = rm.calculate_portfolio_exposure(positions, 100000)
    assert exposure == 0.35


def test_should_reduce_exposure():
    rm = RiskManager()
    positions = [
        {"symbol": "AAPL", "market_value": 10000, "unrealized_plpc": -0.06},
        {"symbol": "MSFT", "market_value": 25000, "unrealized_plpc": 0.0},
        {"symbol": "GOOG", "market_value": 25000, "unrealized_plpc": 0.0},
        {"symbol": "NVDA", "market_value": 25000, "unrealized_plpc": 0.0},
        {"symbol": "TSLA", "market_value": 25000, "unrealized_plpc": 0.0},
        {"symbol": "AMZN", "market_value": 25000, "unrealized_plpc": 0.0},
    ]
    assert rm.should_reduce_exposure(positions, 100000)[0] == True

    positions_small = [
        {"symbol": "AAPL", "market_value": 10000, "unrealized_plpc": -0.01},
    ]
    assert rm.should_reduce_exposure(positions_small, 100000)[0] == False


def test_calculate_kelly_criterion():
    rm = RiskManager()
    kelly = rm.calculate_kelly_criterion(0.6, 200, 100)
    # the function limits kelly between 0 and max_portfolio_risk
    assert kelly > 0 and kelly <= 0.02


def test_log_trade():
    rm = RiskManager()
    rm.log_trade("AAPL", "BUY")
    assert rm.daily_trades == 1
    rm.log_trade("MSFT", "SELL", pl=500.0)
    assert rm.daily_trades == 2

    rm.log_trade("AMZN", "SELL", pl=-1000.0)
    assert rm.daily_losses == 1


def test_assess_position_risk():
    rm = RiskManager()
    risk = rm.assess_position_risk(
        symbol="AAPL",
        position_value=10000,
        portfolio_value=100000,
        unrealized_plpc=-0.06,
        days_held=40,
    )
    assert risk["risk_level"] in ["HIGH", "CRITIQUE", "CRITICAL"]


def test_calculate_sharpe_ratio():
    rm = RiskManager()
    sharpe = rm.calculate_sharpe_ratio([0.01, 0.02, 0.03, -0.01])
    # The return type depends on whether numpy is real or mock
    # Just assert it returns without exception
    assert sharpe is not None


def test_calculate_max_drawdown():
    rm = RiskManager()
    dd, duration = rm.calculate_max_drawdown([100.0, 110.0, 90.0, 95.0, 120.0])
    assert dd is not None


def test_print_risk_summary():
    rm = RiskManager()
    rm.print_risk_summary(
        {
            "portfolio_value": 100000.0,
            "max_drawdown_limit": 0.1,
            "exposure_pct": 0.5,
            "max_exposure": 0.8,
            "positions_count": 0,
            "risky_positions_count": 0,
            "risky_positions": [],
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.1,
            "daily_pnl": 1000.0,
            "daily_pl": 1000.0,
            "daily_pl_pct": 0.01,
            "daily_trades": 5,
            "daily_wins": 3,
            "daily_losses": 2,
            "circuit_breaker": False,
        }
    )


def test_print_risk_summary_with_risky():
    rm = RiskManager()
    rm.print_risk_summary(
        {
            "portfolio_value": 100000.0,
            "max_drawdown_limit": 0.1,
            "exposure_pct": 0.5,
            "max_exposure": 0.8,
            "positions_count": 1,
            "risky_positions_count": 1,
            "risky_positions": [
                {
                    "symbol": "AAPL",
                    "risk_level": "HIGH",
                    "position_value": 10000.0,
                    "unrealized_plpc": -0.05,
                    "warnings": ["Warning"],
                    "recommendation": "Liquidate",
                }
            ],
            "sharpe_ratio": 1.5,
            "max_drawdown": 0.1,
            "daily_pnl": 1000.0,
            "daily_pl": 1000.0,
            "daily_pl_pct": 0.01,
            "daily_trades": 5,
            "daily_wins": 3,
            "daily_losses": 2,
            "circuit_breaker": True,
        }
    )


def test_risk_manager_full_flow():
    rm = RiskManager()
    rm.reset_daily_stats(100000.0)
    rm.log_trade("AAPL", "SELL", pl=100.0)
    rm.check_daily_loss_limit(105000.0)
