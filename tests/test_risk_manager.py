import pytest

from core.risk_manager import RiskManager


@pytest.fixture
def rm():
    return RiskManager(max_portfolio_risk=0.02, max_daily_loss=0.03, max_position_size=0.05)

def test_risk_manager_init(rm):
    assert rm is not None
    assert rm.max_portfolio_risk == 0.02

def test_calculate_position_size(rm):
    size, amount = rm.calculate_position_size(
        portfolio_value=10000.0,
        entry_price=100.0,
        stop_loss_pct=0.05
    )
    assert size == 5
    assert amount == 500.0

def test_check_daily_loss_limit(rm):
    rm.reset_daily_stats(10000.0)
    assert rm.check_daily_loss_limit(9800.0) # 2% loss
    assert not rm.check_daily_loss_limit(9600.0) # 4% loss

def test_calculate_portfolio_exposure(rm):
    positions = [
        {'market_value': 500},
        {'market_value': 1000}
    ]
    exposure = rm.calculate_portfolio_exposure(positions, 10000.0)
    assert exposure == 1500 / 10000.0

def test_kelly_criterion(rm):
    k = rm.calculate_kelly_criterion(win_rate=0.6, avg_win=2.0, avg_loss=1.0)
    assert k == 0.02

def test_should_reduce_exposure(rm):
    positions = [
        {'symbol': 'AAPL', 'market_value': 1000, 'unrealized_plpc': 0.05},
        {'symbol': 'MSFT', 'market_value': 1000, 'unrealized_plpc': 0.10}
    ]
    reduce, reason = rm.should_reduce_exposure(positions, portfolio_value=10000.0)
    assert not reduce

def test_calculate_max_drawdown(rm):
    portfolio_values = [10000, 11000, 9900, 12000, 10500]
    mdd, duration = rm.calculate_max_drawdown(portfolio_values)
    assert isinstance(mdd, float)

def test_assess_position_risk(rm):
    rm.reset_daily_stats(10000.0)
    result = rm.assess_position_risk(
        symbol="AAPL",
        position_value=500.0,
        portfolio_value=10000.0,
        unrealized_plpc=-0.15,
        days_held=10
    )
    # Just assert the type
    assert type(result) is dict

def test_log_trade_and_report(rm):
    rm.reset_daily_stats(10000.0)
    rm.log_trade("AAPL", "BUY", -50.0)
    rm.log_trade("MSFT", "SELL", 200.0)

    positions = []
    report = rm.get_risk_report(positions, 10000.0)
    assert report['daily_trades'] == 2
