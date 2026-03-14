import pytest
from core.risk_manager import RiskManager

def test_risk_manager_initialization():
    rm = RiskManager(
        max_portfolio_risk=0.02,
        max_position_size=0.10,
        max_daily_loss=0.05,
        max_correlation=0.7
    )
    assert rm.max_portfolio_risk == 0.02
    assert rm.max_position_size == 0.10
    assert rm.max_daily_loss == 0.05
    assert rm.max_correlation == 0.7
    assert not rm.circuit_breaker_triggered

def test_calculate_position_size():
    rm = RiskManager(max_position_size=0.10, max_portfolio_risk=0.02)

    shares, position_value = rm.calculate_position_size(
        portfolio_value=100000.0,
        entry_price=100.0,
        stop_loss_pct=0.10
    )
    # Capped by 10% of portfolio value => $10,000 => 100 shares
    assert shares == 100
    assert position_value == 10000.0

def test_calculate_portfolio_exposure():
    rm = RiskManager()
    positions = [
        {'symbol': 'AAPL', 'market_value': 10000.0},
        {'symbol': 'MSFT', 'market_value': 20000.0}
    ]
    exposure = rm.calculate_portfolio_exposure(positions, 100000.0)
    assert exposure == 0.30

def test_should_reduce_exposure():
    rm = RiskManager(max_position_size=0.10)

    positions = [
        {'symbol': 'AAPL', 'market_value': 5000.0, 'unrealized_plpc': 0.05},
        {'symbol': 'MSFT', 'market_value': 5000.0, 'unrealized_plpc': 0.05}
    ]
    reduce, msg = rm.should_reduce_exposure(positions, 100000.0)
    assert reduce is False

    positions_losing = [
        {'symbol': 'AAPL', 'market_value': 5000.0, 'unrealized_plpc': -0.10},
        {'symbol': 'MSFT', 'market_value': 5000.0, 'unrealized_plpc': -0.10}
    ]
    reduce, msg = rm.should_reduce_exposure(positions_losing, 100000.0)
    assert reduce is True

def test_check_daily_loss_limit_and_reset():
    rm = RiskManager(max_daily_loss=0.05)

    # Initial value set
    rm.reset_daily_stats(100000.0)
    assert rm.daily_start_value == 100000.0

    # No circuit breaker
    rm.check_daily_loss_limit(96000.0) # -4%
    assert not rm.circuit_breaker_triggered

    # Circuit breaker triggered
    rm.check_daily_loss_limit(94000.0) # -6%
    assert rm.circuit_breaker_triggered

def test_kelly_criterion():
    rm = RiskManager(max_portfolio_risk=0.10)

    kelly = rm.calculate_kelly_criterion(win_rate=0.5, avg_win=0.10, avg_loss=0.05)
    assert kelly == 0.10

    kelly = rm.calculate_kelly_criterion(win_rate=0.4, avg_win=0.05, avg_loss=0.05)
    assert kelly == 0.0

def test_calculate_sharpe_ratio():
    rm = RiskManager()
    returns = [0.01, -0.005, 0.02, 0.005, -0.01]
    sharpe = rm.calculate_sharpe_ratio(returns, risk_free_rate=0.0)
    assert sharpe > 0

def test_calculate_max_drawdown():
    rm = RiskManager()
    values = [100, 110, 99, 120, 90, 100]
    dd_pct, dd_amount = rm.calculate_max_drawdown(values)
    assert dd_amount == 30
    assert dd_pct == pytest.approx(-30 / 120)

def test_assess_position_risk():
    rm = RiskManager(max_position_size=0.10)

    risk = rm.assess_position_risk(
        symbol='AAPL',
        position_value=15000.0,
        portfolio_value=100000.0,
        unrealized_plpc=-0.15,
        days_held=40
    )
    assert risk['risk_level'] == 'CRITIQUE'
    assert risk['risk_score'] >= 4

    risk = rm.assess_position_risk(
        symbol='MSFT',
        position_value=5000.0,
        portfolio_value=100000.0,
        unrealized_plpc=0.05,
        days_held=10
    )
    assert risk['risk_level'] == 'FAIBLE'
    assert risk['risk_score'] == 0

def test_get_risk_report():
    rm = RiskManager()
    positions = [
        {'symbol': 'AAPL', 'market_value': 5000.0, 'unrealized_plpc': -0.15, 'purchase_date': '2023-01-01T00:00:00Z'}
    ]
    report = rm.get_risk_report(positions, 100000.0, daily_returns=[0.01, -0.01, 0.02])
    assert report['portfolio_value'] == 100000.0
    assert report['exposure_pct'] == 0.05
    assert report['risky_positions_count'] == 1

def test_log_trade():
    rm = RiskManager()
    rm.log_trade('AAPL', 'buy', pl=100.0)
    rm.log_trade('MSFT', 'sell', pl=-50.0)
    rm.log_trade('GOOG', 'buy')

    assert rm.daily_trades == 3
    assert rm.daily_wins == 1
    assert rm.daily_losses == 1
