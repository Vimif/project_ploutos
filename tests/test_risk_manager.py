import unittest
from unittest.mock import MagicMock
from core.risk_manager import RiskManager

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        # Correct arguments for RiskManager __init__ based on code
        self.risk_manager = RiskManager(
            max_portfolio_risk=0.02,
            max_daily_loss=0.03,
            max_position_size=0.05,
            max_correlation=0.7
        )
        # Manually set daily_start_value to avoid None checks failing
        self.risk_manager.daily_start_value = 10000.0

    def test_initialization(self):
        self.assertEqual(self.risk_manager.max_portfolio_risk, 0.02)
        self.assertEqual(self.risk_manager.max_daily_loss, 0.03)
        self.assertEqual(self.risk_manager.max_position_size, 0.05)
        self.assertEqual(self.risk_manager.max_correlation, 0.7)

    def test_calculate_position_size(self):
        # The logic in RiskManager reduces position size if it exceeds max_position_size (5%)
        # In previous fail:
        # Risk amount = 10000 * 0.01 = 100
        # Stop loss distance = 100 * 0.10 = 10
        # Quantity = 100 / 10 = 10
        # Position Value = 10 * 100 = 1000
        # Position Pct = 1000 / 10000 = 10%
        # Max position pct is 5%, so it gets capped.
        # Max position value = 10000 * 0.05 = 500
        # New quantity = 500 / 100 = 5

        qty, val = self.risk_manager.calculate_position_size(
            portfolio_value=10000,
            entry_price=100,
            stop_loss_pct=0.10, # 10% stop loss
            risk_pct=0.01 # 1% risk
        )

        self.assertEqual(qty, 5)
        self.assertEqual(val, 500)

    def test_check_daily_loss_limit_pass(self):
        # Daily start = 10000
        # Current = 9800 (2% loss)
        # Max loss = 3%
        self.assertTrue(self.risk_manager.check_daily_loss_limit(9800))

    def test_check_daily_loss_limit_fail(self):
        # Daily start = 10000
        # Current = 9600 (4% loss)
        # Max loss = 3%
        self.assertFalse(self.risk_manager.check_daily_loss_limit(9600))
        self.assertTrue(self.risk_manager.circuit_breaker_triggered)

    def test_reset_daily_stats(self):
        self.risk_manager.circuit_breaker_triggered = True
        self.risk_manager.daily_trades = 10
        self.risk_manager.reset_daily_stats(12000)

        self.assertEqual(self.risk_manager.daily_start_value, 12000)
        self.assertFalse(self.risk_manager.circuit_breaker_triggered)
        self.assertEqual(self.risk_manager.daily_trades, 0)

if __name__ == "__main__":
    unittest.main()
