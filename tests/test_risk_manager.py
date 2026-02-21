import unittest

from core.risk_manager import RiskManager


class TestRiskManager(unittest.TestCase):
    def setUp(self):
        # RiskManager does not take initial_balance, it tracks daily_start_value
        self.rm = RiskManager()
        self.rm.reset_daily_stats(10000.0)

    def test_calculate_position_size(self):
        # Test default
        # calculate_position_size(portfolio_value, entry_price, stop_loss_pct, risk_pct)
        # return (quantity, position_value)
        qty, val = self.rm.calculate_position_size(10000.0, 150.0, 0.05)
        self.assertIsInstance(qty, int)
        self.assertIsInstance(val, float)

        # Test max position size cap
        self.rm.max_position_size = 0.1  # 10%
        qty, val = self.rm.calculate_position_size(10000.0, 100.0, 0.05)
        # Max pos value = 1000. Price = 100. Qty = 10.
        self.assertEqual(qty, 10)

    def test_check_daily_loss_limit(self):
        # Initial 10000. Limit 3% (300).
        # Value 9600 -> Loss 400 (>300) -> Fail
        self.assertFalse(self.rm.check_daily_loss_limit(9600.0))

        # Value 9800 -> Loss 200 (<300) -> Pass
        self.assertTrue(self.rm.check_daily_loss_limit(9800.0))

    def test_log_trade(self):
        self.rm.log_trade("AAPL", "BUY", 500)
        self.assertEqual(self.rm.daily_wins, 1)
        self.assertEqual(self.rm.daily_trades, 1)

    def test_circuit_breaker(self):
        # Initial is 10000. Limit is 3% (300).
        # Loss of 400 should trigger.
        current_value = 9600.0
        can_trade = self.rm.check_daily_loss_limit(current_value)
        self.assertFalse(can_trade)
        self.assertTrue(self.rm.circuit_breaker_triggered)


if __name__ == "__main__":
    unittest.main()
