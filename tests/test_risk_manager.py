import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from core.risk_manager import RiskManager

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        # Pass parameters directly instead of config dict
        self.risk_manager = RiskManager(
            max_portfolio_risk=0.02,
            max_daily_loss=0.03,
            max_position_size=0.05,
            max_correlation=0.7
        )

    def test_initialization(self):
        self.assertEqual(self.risk_manager.max_position_size, 0.05)
        self.assertEqual(self.risk_manager.max_portfolio_risk, 0.02)

    def test_calculate_position_size(self):
        # Portfolio value 100k, price 100
        # risk per trade 2% = 2000
        # stop loss 5% = 5 distance
        # qty = 2000 / 5 = 400
        # value = 400 * 100 = 40000
        # max position size = 5% of 100k = 5000
        # Expected: capped at 5000 / 100 = 50 shares

        qty, value = self.risk_manager.calculate_position_size(100000, 100, 0.05, 0.02)

        # Max position value is 5000
        expected_value = 5000
        expected_qty = 50

        self.assertEqual(qty, expected_qty)
        self.assertEqual(value, expected_value)

    def test_update_daily_stats(self):
        # Manually set daily_trades via log_trade
        self.risk_manager.log_trade("AAPL", "BUY")
        self.assertEqual(self.risk_manager.daily_trades, 1)

    def test_check_daily_loss_limit(self):
        # Set start value
        self.risk_manager.daily_start_value = 100000

        # Current value 96000 (-4%) -> Should trigger limit (3%)
        allowed = self.risk_manager.check_daily_loss_limit(96000)
        self.assertFalse(allowed)
        self.assertTrue(self.risk_manager.circuit_breaker_triggered)

        # Verify subsequent calls return False
        allowed = self.risk_manager.check_daily_loss_limit(105000) # Recovery
        self.assertFalse(allowed)

    def test_reset_daily_stats(self):
        self.risk_manager.daily_trades = 5
        self.risk_manager.circuit_breaker_triggered = True
        self.risk_manager.reset_daily_stats(100000)

        self.assertEqual(self.risk_manager.daily_trades, 0)
        self.assertFalse(self.risk_manager.circuit_breaker_triggered)
        self.assertEqual(self.risk_manager.daily_start_value, 100000)

if __name__ == '__main__':
    unittest.main()
