import unittest

from core.risk_manager import RiskManager


class TestRiskManager(unittest.TestCase):
    def setUp(self):
        # Fix constructor: remove initial_balance
        self.risk_manager = RiskManager(
            max_portfolio_risk=0.02, max_daily_loss=0.03, max_position_size=0.05
        )
        self.risk_manager.daily_start_value = 10000.0

    def test_initialization(self):
        self.assertEqual(self.risk_manager.max_portfolio_risk, 0.02)
        # Check initial state
        self.assertEqual(self.risk_manager.daily_trades, 0)

    def test_calculate_position_size(self):
        # Fix arguments: portfolio_value, entry_price, stop_loss_pct
        quantity, value = self.risk_manager.calculate_position_size(
            portfolio_value=10000.0, entry_price=150.0, stop_loss_pct=0.05  # 5% stop loss
        )

        # Risk amount = 10000 * 0.02 = 200
        # Stop distance = 150 * 0.05 = 7.5
        # Quantity = 200 / 7.5 = 26.66 -> 26
        # Value = 26 * 150 = 3900

        # BUT max_position_size = 0.05 -> 500
        # So it should be capped at 500
        # 500 / 150 = 3.33 -> 3

        self.assertLessEqual(value, 10000.0 * 0.05 + 150)  # Approx check
        self.assertGreater(quantity, 0)

    def test_check_daily_loss_limit(self):
        # Start value is 10000
        # Current value 9000 -> 10% loss -> should trigger (limit is 3%)
        allowed = self.risk_manager.check_daily_loss_limit(9000.0)
        self.assertFalse(allowed)
        self.assertTrue(self.risk_manager.circuit_breaker_triggered)

        # Reset
        self.risk_manager.reset_daily_stats(10000.0)
        allowed = self.risk_manager.check_daily_loss_limit(10100.0)
        self.assertTrue(allowed)

    def test_assess_position_risk(self):
        risk = self.risk_manager.assess_position_risk(
            symbol="AAPL",
            position_value=1000.0,
            portfolio_value=10000.0,
            unrealized_plpc=-0.15,  # High loss
            days_held=10,
        )
        self.assertEqual(
            risk["risk_level"], "CRITIQUE"
        )  # >10% loss -> score 3, + position 10% (max 5%) -> score 2 = 5
