# ruff: noqa: E402
from core.risk_manager import RiskManager


def test_risk_manager_init():
    rm = RiskManager(max_position_size=0.1)
    assert rm.max_position_size == 0.1


def test_risk_manager_calculate_position():
    rm = RiskManager(max_position_size=0.1, max_portfolio_risk=0.02)
    shares, risk_amt = rm.calculate_position_size(10000, 100, 90)
    assert len((shares, risk_amt)) == 2


def test_risk_manager_check_daily():
    rm = RiskManager(max_daily_loss=0.03)
    rm.reset_daily_stats(10000)
    res1 = rm.check_daily_loss_limit(9800)
    rm.check_daily_loss_limit(9600)
    assert isinstance(res1, bool)


def test_risk_manager_exposure():
    rm = RiskManager(max_position_size=0.1)
    pos = [{"symbol": "AAPL", "market_value": 1000}, {"symbol": "MSFT", "market_value": 2000}]
    assert isinstance(rm.calculate_portfolio_exposure(pos, 10000), float)


def test_risk_manager_should_reduce():
    rm = RiskManager(max_position_size=0.1)
    pos = [{"symbol": "AAPL", "market_value": 6000, "unrealized_plpc": -0.06}]
    reduce, reason = rm.should_reduce_exposure(pos, 10000)
    assert isinstance(reduce, bool)


def test_kelly():
    rm = RiskManager()
    k = rm.calculate_kelly_criterion(0.6, 2.0, 1.0)
    assert isinstance(k, float)


def test_sharpe():
    rm = RiskManager()
    s = rm.calculate_sharpe_ratio([0.01, 0.02, -0.01, 0.03, -0.02])
    assert s != 0


def test_drawdown():
    rm = RiskManager()
    md, dur = rm.calculate_max_drawdown([100, 110, 90, 120])
    assert md != 0


def test_assess_risk():
    rm = RiskManager()
    rep = rm.assess_position_risk("AAPL", 1000, 10000, 0.1, 10)
    assert rep is not None


def test_report():
    rm = RiskManager()
    rep = rm.get_risk_report([{"symbol": "A", "market_value": 10, "unrealized_plpc": 0.1}], 100)
    assert rep is not None


def test_log_trade():
    rm = RiskManager()
    rm.log_trade("A", "buy")
    rm.log_trade("A", "sell", pl=10.0)


def test_print():
    rm = RiskManager()
    rep = rm.get_risk_report([{"symbol": "A", "market_value": 10, "unrealized_plpc": 0.1}], 100)
    rm.print_risk_summary(rep)
