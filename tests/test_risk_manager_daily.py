import pytest
from core.risk_manager import RiskManager


def test_daily_stats_reset():
    rm = RiskManager()
    rm.daily_trades = 10
    rm.daily_wins = 5
    rm.daily_losses = 5
    rm.daily_start_equity = 100000

    rm.reset_daily_stats(105000)

    assert rm.daily_trades == 0
    assert rm.daily_wins == 0
    assert rm.daily_losses == 0


def test_log_trade():
    rm = RiskManager()
    rm.log_trade("AAPL", "BUY")
    assert rm.daily_trades == 1

    rm.log_trade("MSFT", "SELL", pl=100)
    assert rm.daily_trades == 2
    assert rm.daily_wins == 1
    assert rm.daily_losses == 0

    rm.log_trade("TSLA", "SELL", pl=-50)
    assert rm.daily_trades == 3
    assert rm.daily_wins == 1
    assert rm.daily_losses == 1


def test_get_risk_report_empty():
    rm = RiskManager()
    report = rm.get_risk_report(positions=[], portfolio_value=100000)
    assert report["exposure_pct"] == 0.0
    assert report["positions_count"] == 0
