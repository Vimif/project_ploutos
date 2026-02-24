
import unittest

from core.risk_manager import RiskManager


class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.risk_manager = RiskManager(
            max_portfolio_risk=0.02,
            max_daily_loss=0.03,
            max_position_size=0.05
        )
        self.risk_manager.daily_start_value = 10000.0

    def test_initialization(self):
        self.assertEqual(self.risk_manager.max_portfolio_risk, 0.02)
        self.assertEqual(self.risk_manager.daily_trades, 0)

    def test_calculate_position_size(self):
        quantity, value = self.risk_manager.calculate_position_size(
            portfolio_value=10000.0,
            entry_price=150.0,
            stop_loss_pct=0.05
        )
        self.assertGreater(quantity, 0)

        # Test max position size cap
        # Risk 2% = 200
        # Stop 1% = 1.5
        # Qty = 200/1.5 = 133
        # Value = 133*150 = 19950 > 500 (5% of 10000)
        # Should be capped
        quantity_capped, value_capped = self.risk_manager.calculate_position_size(
            portfolio_value=10000.0,
            entry_price=150.0,
            stop_loss_pct=0.01
        )
        self.assertLessEqual(value_capped, 500.0 + 150.0) # allow rounding margin

    def test_check_daily_loss_limit(self):
        # No daily start value yet
        rm = RiskManager()
        self.assertTrue(rm.check_daily_loss_limit(10000.0))
        self.assertEqual(rm.daily_start_value, 10000.0)

        # Normal fluctuation
        self.assertTrue(rm.check_daily_loss_limit(9900.0)) # -1%

        # Limit hit
        self.assertFalse(rm.check_daily_loss_limit(9000.0)) # -10% > -3%
        self.assertTrue(rm.circuit_breaker_triggered)

    def test_reset_daily_stats(self):
        self.risk_manager.log_trade("AAPL", "BUY", 100.0)
        self.risk_manager.check_daily_loss_limit(9000.0) # Trigger breaker

        self.risk_manager.reset_daily_stats(11000.0)
        self.assertEqual(self.risk_manager.daily_start_value, 11000.0)
        self.assertEqual(self.risk_manager.daily_trades, 0)
        self.assertFalse(self.risk_manager.circuit_breaker_triggered)

    def test_calculate_portfolio_exposure(self):
        positions = [
            {"market_value": 1000.0},
            {"market_value": 2000.0}
        ]
        exposure = self.risk_manager.calculate_portfolio_exposure(positions, 10000.0)
        self.assertEqual(exposure, 0.3)

    def test_should_reduce_exposure(self):
        # Low exposure
        positions = [{"market_value": 1000.0, "unrealized_plpc": 0.01}]
        reduce, reason = self.risk_manager.should_reduce_exposure(positions, 10000.0)
        self.assertFalse(reduce)

        # High exposure
        positions_high = [{"market_value": 9000.0, "unrealized_plpc": 0.01}]
        reduce, reason = self.risk_manager.should_reduce_exposure(positions_high, 10000.0)
        self.assertTrue(reduce)
        self.assertIn("Exposition élevée", reason)

        # Many losers
        positions_losers = [
            {"market_value": 100.0, "unrealized_plpc": -0.1},
            {"market_value": 100.0, "unrealized_plpc": -0.1},
            {"market_value": 100.0, "unrealized_plpc": 0.1}
        ]
        reduce, reason = self.risk_manager.should_reduce_exposure(positions_losers, 10000.0)
        self.assertTrue(reduce)
        self.assertIn("positions en perte", reason)

    def test_calculate_kelly_criterion(self):
        # 50% win rate, 2:1 reward/risk
        # K = 0.5 - (0.5 / 2) = 0.5 - 0.25 = 0.25
        # Half Kelly = 0.125
        # Capped at max_portfolio_risk (0.02)
        k = self.risk_manager.calculate_kelly_criterion(0.5, 0.2, 0.1)
        self.assertEqual(k, 0.02)

        # Bad stats
        k = self.risk_manager.calculate_kelly_criterion(0.0, 0.1, 0.1)
        self.assertEqual(k, 0.02) # Fallback

    def test_calculate_sharpe_ratio(self):
        returns = [0.01, -0.01, 0.02, 0.0]
        sharpe = self.risk_manager.calculate_sharpe_ratio(returns)
        self.assertIsInstance(sharpe, float)

        # Not enough data
        self.assertEqual(self.risk_manager.calculate_sharpe_ratio([]), 0.0)

    def test_calculate_max_drawdown(self):
        values = [100, 110, 100, 90, 120]
        # Peak 110, Low 90 -> DD = (110-90)/110 = 20/110 = 0.18
        pct, amount = self.risk_manager.calculate_max_drawdown(values)
        self.assertAlmostEqual(pct, -0.1818, places=4)
        self.assertEqual(amount, 20)

    def test_assess_position_risk(self):
        # Risky
        risk = self.risk_manager.assess_position_risk(
            "AAPL", 600.0, 10000.0, -0.15, 40
        )
        self.assertGreaterEqual(risk["risk_score"], 4)
        self.assertEqual(risk["recommendation"], "FERMER IMMÉDIATEMENT")

        # Safe
        risk = self.risk_manager.assess_position_risk(
            "MSFT", 400.0, 10000.0, 0.05, 5
        )
        self.assertEqual(risk["risk_level"], "FAIBLE")

    def test_get_risk_report(self):
        positions = [{"symbol": "AAPL", "market_value": 1000, "unrealized_plpc": 0.0}]
        report = self.risk_manager.get_risk_report(positions, 10000.0)
        self.assertEqual(report["portfolio_value"], 10000.0)
        self.assertFalse(report["circuit_breaker"])

    def test_log_trade(self):
        self.risk_manager.log_trade("AAPL", "BUY", 100)
        self.assertEqual(self.risk_manager.daily_wins, 1)
        self.risk_manager.log_trade("AAPL", "SELL", -50)
        self.assertEqual(self.risk_manager.daily_losses, 1)
        self.assertEqual(self.risk_manager.daily_trades, 2)
