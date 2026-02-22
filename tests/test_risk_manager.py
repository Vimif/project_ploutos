import pytest

from core.risk_manager import RiskManager


@pytest.fixture
def risk_manager():
    return RiskManager(
        max_portfolio_risk=0.02,
        max_daily_loss=0.03,
        max_position_size=0.1,  # 10%
        max_correlation=0.7,
    )


def test_get_risk_report(risk_manager):
    positions = [
        {
            "symbol": "AAPL",
            "market_value": 10000,
            "unrealized_plpc": 0.05,
            "purchase_date": "2023-01-01T10:00:00",
        },
        {
            "symbol": "TSLA",
            "market_value": 5000,
            "unrealized_plpc": -0.15,  # Loss > 10% -> Critical
            "purchase_date": "2023-01-01T10:00:00",
        },
    ]
    portfolio_value = 100000
    daily_returns = [0.01, 0.02, -0.01]

    report = risk_manager.get_risk_report(positions, portfolio_value, daily_returns)

    assert report["portfolio_value"] == 100000
    assert report["positions_count"] == 2
    assert report["exposure_pct"] == 0.15
    assert report["risky_positions_count"] >= 1
    assert any(p["symbol"] == "TSLA" for p in report["risky_positions"])
    assert report["sharpe_ratio"] > 0


def test_log_trade(risk_manager):
    risk_manager.log_trade("AAPL", "BUY")
    assert risk_manager.daily_trades == 1

    risk_manager.log_trade("AAPL", "SELL", pl=100)
    assert risk_manager.daily_trades == 2
    assert risk_manager.daily_wins == 1

    risk_manager.log_trade("TSLA", "SELL", pl=-50)
    assert risk_manager.daily_trades == 3
    assert risk_manager.daily_losses == 1
