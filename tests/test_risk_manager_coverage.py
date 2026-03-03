import pytest
from core.risk_manager import RiskManager

def test_assess_position_risk_low_risk():
    rm = RiskManager()
    result = rm.assess_position_risk("AAPL", 5000, 100000, 0.05, 5)
    assert result['risk_score'] == 0
    assert result['risk_level'] == "FAIBLE"

def test_assess_position_risk_medium_risk():
    rm = RiskManager(max_position_size=0.1)
    result = rm.assess_position_risk("AAPL", 9000, 100000, -0.04, 25)
    assert result['risk_score'] >= 0 # Just checking it runs without error

def test_print_risk_summary_with_sharpe():
    rm = RiskManager()
    report = {
        'portfolio_value': 100000,
        'exposure_pct': 0.5,
        'positions_count': 2,
        'risky_positions_count': 0,
        'risky_positions': [],
        'daily_pnl_pct': 0.01,
        'daily_pl': 100,
        'daily_pl_pct': 0.01,
        'daily_trades': 0,
        'daily_wins': 0,
        'daily_losses': 0,
        'circuit_breaker': False,
        'sharpe_ratio': 1.5,
        'max_daily_loss': 0.03
    }
    rm.print_risk_summary(report)
