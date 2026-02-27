import pytest
import numpy as np
from core.risk_manager import RiskManager

def test_risk_manager_coverage():
    rm = RiskManager(
        max_portfolio_risk=0.02,
        max_daily_loss=0.03,
        max_position_size=0.05,
        max_correlation=0.7
    )

    # 1. Position Sizing
    qty, val = rm.calculate_position_size(
        portfolio_value=100000,
        entry_price=100,
        stop_loss_pct=0.05,
        risk_pct=0.01
    )
    # Risk = 1000, Stop dist = 5. Qty = 200. Val = 20000 (20%)
    # Max pos size is 5% (5000). Qty capped at 5000/100 = 50.
    assert val <= 5000 + 100 # buffer for float errors
    assert qty <= 50

    # 2. Daily Loss Limit (Circuit Breaker)
    rm.reset_daily_stats(100000)
    assert rm.check_daily_loss_limit(98000) # -2% -> OK
    assert not rm.check_daily_loss_limit(96000) # -4% -> CB Triggered
    assert rm.circuit_breaker_triggered

    # 3. Portfolio Exposure
    positions = [
        {"symbol": "A", "market_value": 10000, "unrealized_plpc": 0.1},
        {"symbol": "B", "market_value": 80000, "unrealized_plpc": -0.1}
    ]
    exposure = rm.calculate_portfolio_exposure(positions, 100000)
    assert exposure == 0.9

    should_reduce, reason = rm.should_reduce_exposure(positions, 100000)
    assert should_reduce
    assert "Exposition élevée" in reason

    # 4. Kelly Criterion
    # W=0.5, R=2 (20%/10%) -> K = 0.5 - (0.5/2) = 0.25 -> Half = 0.125 -> Capped at 0.02
    kelly = rm.calculate_kelly_criterion(0.5, 0.2, 0.1)
    assert kelly == 0.02

    # 5. Sharpe Ratio
    returns = [0.01, -0.01, 0.02, 0.0]
    sharpe = rm.calculate_sharpe_ratio(returns)
    assert sharpe > 0

    # 6. Max Drawdown
    values = [100, 110, 99, 105]
    dd_pct, dd_amt = rm.calculate_max_drawdown(values)
    # Peak 110, Valley 99 -> DD 11/110 = 0.1
    assert abs(dd_pct - (-0.1)) < 1e-6
    assert dd_amt == 11

    # 7. Risk Report
    # Create fake position with date
    positions[0]["purchase_date"] = "2023-01-01T12:00:00"
    report = rm.get_risk_report(positions, 100000, daily_returns=[0.01, -0.01])
    assert report["exposure_pct"] == 0.9
    assert report["circuit_breaker"] == True # from step 2

    # 8. Print Summary
    rm.print_risk_summary(report)

    # 9. Log Trade
    rm.log_trade("AAPL", "BUY", 100)
    assert rm.daily_trades == 1
    assert rm.daily_wins == 1
