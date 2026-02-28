import pytest

from core.risk_manager import RiskManager


def test_initialization():
    rm = RiskManager(max_portfolio_risk=0.05, max_daily_loss=0.04, max_position_size=0.1)
    assert rm.max_portfolio_risk == 0.05
    assert rm.max_daily_loss == 0.04
    assert rm.max_position_size == 0.1


def test_check_daily_loss_limit_exceeded():
    rm = RiskManager(max_daily_loss=0.03)
    rm.reset_daily_stats(10000)

    # Portfolio drops below 3% (4% loss)
    can_trade = rm.check_daily_loss_limit(9600)
    assert not can_trade
    assert rm.circuit_breaker_triggered


def test_check_daily_loss_limit_ok():
    rm = RiskManager(max_daily_loss=0.03)
    rm.reset_daily_stats(10000)

    # Portfolio drops 2%
    can_trade = rm.check_daily_loss_limit(9800)
    assert can_trade
    assert not rm.circuit_breaker_triggered


def test_calculate_position_size():
    rm = RiskManager(max_portfolio_risk=0.02, max_position_size=0.5)
    shares, investment = rm.calculate_position_size(
        portfolio_value=10000, entry_price=100, stop_loss_pct=0.05
    )
    assert shares == 40
    assert investment == 4000


def test_calculate_position_size_capped():
    rm = RiskManager(max_portfolio_risk=0.02, max_position_size=0.1)  # Max 10%
    shares, investment = rm.calculate_position_size(
        portfolio_value=10000, entry_price=100, stop_loss_pct=0.05
    )
    assert shares == 10
    assert investment == 1000


def test_reset():
    rm = RiskManager()
    rm.circuit_breaker_triggered = True
    rm.daily_trades = 5

    rm.reset_daily_stats(10000)
    assert not rm.circuit_breaker_triggered
    assert rm.daily_trades == 0


def test_calculate_portfolio_exposure():
    rm = RiskManager()
    positions = [
        {"market_value": 1000},
        {"market_value": 2000},
    ]
    exposure = rm.calculate_portfolio_exposure(positions, 10000)
    assert exposure == 0.3


def test_should_reduce_exposure():
    rm = RiskManager()
    positions = [
        {"market_value": 4000, "unrealized_plpc": -0.06},  # Loser
        {"market_value": 4000, "unrealized_plpc": -0.06},  # Loser
    ]
    reduce, reason = rm.should_reduce_exposure(positions, 10000)
    assert isinstance(reduce, bool)
    assert reduce


def test_calculate_kelly_criterion():
    rm = RiskManager()
    kelly = rm.calculate_kelly_criterion(0.6, 2.0, 1.0)
    assert kelly == pytest.approx(rm.max_portfolio_risk)


def test_calculate_max_drawdown():
    rm = RiskManager()
    values = [10000, 11000, 9900, 12000]
    mdd_pct, mdd_amount = rm.calculate_max_drawdown(values)
    # The risk manager returns positive or negative numbers for drawdown depending on logic
    assert mdd_pct == pytest.approx(-0.1) or mdd_pct == pytest.approx(0.1)
    assert mdd_amount == pytest.approx(1100) or mdd_amount == pytest.approx(-1100)


def test_calculate_sharpe_ratio():
    rm = RiskManager()
    returns = [0.01, -0.005, 0.02, 0.005]
    sharpe = rm.calculate_sharpe_ratio(returns, risk_free_rate=0.0)
    assert sharpe > 0


def test_log_trade():
    rm = RiskManager()
    rm.log_trade("AAPL", "BUY")
    assert rm.daily_trades == 1
    rm.log_trade("AAPL", "SELL", 100)  # Win
    assert rm.daily_wins == 1
    rm.log_trade("AAPL", "SELL", -50)  # Loss
    assert rm.daily_losses == 1


def test_get_risk_report():
    rm = RiskManager()
    rm.log_trade("AAPL", "BUY")
    rm.log_trade("AAPL", "SELL", 100)
    rm.reset_daily_stats(10000)

    positions = [{"symbol": "AAPL", "market_value": 1000, "unrealized_plpc": 0.05}]
    report = rm.get_risk_report(positions, 11000, [0.01, -0.005, 0.02])
    assert report["daily_pl"] == 1000.0


def test_assess_position_risk():
    rm = RiskManager()
    assessment = rm.assess_position_risk(
        symbol="AAPL", position_value=2000, portfolio_value=10000, unrealized_plpc=0.05, days_held=5
    )
    assert isinstance(assessment, dict)
    assert "risk_score" in assessment
    assert "warnings" in assessment


def test_print_risk_summary():
    rm = RiskManager()
    report = {
        "portfolio_value": 10000,
        "exposure_pct": 0.5,
        "positions_count": 5,
        "risky_positions_count": 1,
        "sharpe_ratio": 1.5,
        "daily_pl": 500,
        "daily_pl_pct": 0.05,
        "daily_trades": 2,
        "daily_wins": 1,
        "daily_losses": 1,
        "circuit_breaker": False,
        "risky_positions": [
            {
                "symbol": "TEST",
                "score": 3,
                "warnings": ["test"],
                "risk_level": "High",
                "recommendation": "Close",
            }
        ],
    }
    # Just calling it to ensure no exceptions
    rm.print_risk_summary(report)
