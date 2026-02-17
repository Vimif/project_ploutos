import unittest
from datetime import datetime, timezone

# Add project root to path if needed (though pytest usually handles it)
# sys.path.insert(0, str(Path(__file__).parent.parent))
from core.risk_manager import RiskManager


class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.risk_manager = RiskManager(
            max_portfolio_risk=0.02,
            max_daily_loss=0.03,
            max_position_size=0.05,
            max_correlation=0.7,
        )

    def test_initialization(self):
        self.assertEqual(self.risk_manager.max_portfolio_risk, 0.02)
        self.assertEqual(self.risk_manager.max_daily_loss, 0.03)
        self.assertEqual(self.risk_manager.max_position_size, 0.05)
        self.assertEqual(self.risk_manager.max_correlation, 0.7)
        self.assertFalse(self.risk_manager.circuit_breaker_triggered)

    def test_calculate_position_size(self):
        portfolio_value = 100_000
        entry_price = 150.0
        stop_loss_pct = 0.05  # 5% stop loss

        # Risk amount = 100_000 * 0.02 = 2000
        # Stop loss distance = 150 * 0.05 = 7.5
        # Quantity = 2000 / 7.5 = 266.66 -> 266

        qty, value = self.risk_manager.calculate_position_size(
            portfolio_value, entry_price, stop_loss_pct
        )
        self.assertEqual(qty, 266)
        self.assertAlmostEqual(value, 266 * 150.0)

    def test_calculate_position_size_capped(self):
        # Case where position size would exceed max_position_size (5%)
        portfolio_value = 100_000
        entry_price = 100.0
        stop_loss_pct = 0.01  # Very tight stop -> huge position allowed by risk

        # Risk amount = 2000
        # Stop distance = 1.0
        # Calc qty = 2000
        # Value = 200,000 (200% of portfolio!) -> Should be capped at 5% (5000)

        qty, value = self.risk_manager.calculate_position_size(
            portfolio_value, entry_price, stop_loss_pct
        )

        expected_max_value = portfolio_value * 0.05
        expected_qty = int(expected_max_value / entry_price)

        self.assertEqual(qty, expected_qty)
        self.assertLessEqual(value, expected_max_value)

    def test_check_daily_loss_limit(self):
        # Start day
        self.risk_manager.reset_daily_stats(100_000)

        # Current value 98,000 (-2%) -> OK
        self.assertTrue(self.risk_manager.check_daily_loss_limit(98_000))
        self.assertFalse(self.risk_manager.circuit_breaker_triggered)

        # Current value 96,000 (-4%) -> FAIL (max -3%)
        self.assertFalse(self.risk_manager.check_daily_loss_limit(96_000))
        self.assertTrue(self.risk_manager.circuit_breaker_triggered)

    def test_reset_daily_stats(self):
        self.risk_manager.circuit_breaker_triggered = True
        self.risk_manager.reset_daily_stats(105_000)
        self.assertFalse(self.risk_manager.circuit_breaker_triggered)
        self.assertEqual(self.risk_manager.daily_start_value, 105_000)

    def test_calculate_portfolio_exposure(self):
        positions = [{"market_value": 10_000}, {"market_value": 20_000}]
        portfolio_value = 100_000

        exposure = self.risk_manager.calculate_portfolio_exposure(positions, portfolio_value)
        self.assertEqual(exposure, 0.3)

    def test_should_reduce_exposure_high_exposure(self):
        positions = [{"market_value": 90_000, "unrealized_plpc": 0.0}]
        portfolio_value = 100_000

        reduce, reason = self.risk_manager.should_reduce_exposure(positions, portfolio_value)
        self.assertTrue(reduce)
        self.assertIn("Exposition élevée", reason)

    def test_should_reduce_exposure_losing_positions(self):
        positions = [
            {"market_value": 1000, "unrealized_plpc": -0.10},  # Loss > 5%
            {"market_value": 1000, "unrealized_plpc": -0.06},  # Loss > 5%
            {"market_value": 1000, "unrealized_plpc": 0.01},
        ]
        # 2/3 losing > 60%
        portfolio_value = 100_000

        reduce, reason = self.risk_manager.should_reduce_exposure(positions, portfolio_value)
        self.assertTrue(reduce)
        self.assertIn("positions en perte", reason)

    def test_calculate_kelly_criterion(self):
        # Win rate 50%, Win/Loss 2:1
        # Kelly = 0.5 - (0.5 / 2) = 0.5 - 0.25 = 0.25
        # Half Kelly = 0.125
        # Capped at max_risk (0.02)

        kelly = self.risk_manager.calculate_kelly_criterion(0.5, 0.2, 0.1)
        self.assertEqual(kelly, 0.02)

        # Test with very small numbers to not cap
        # Win 60%, R=1
        # K = 0.6 - 0.4/1 = 0.2
        # Half = 0.1
        # Capped at 0.02

        # We need a case where half kelly < max risk (0.02)
        # K = 0.03 -> Half = 0.015 -> Used 0.015
        # 0.03 = W - (1-W)/R
        # Let R=100 (huge win ratio), W=0.03
        # K ~ 0.03

        kelly_small = self.risk_manager.calculate_kelly_criterion(0.51, 0.02, 0.019)
        # Just verifying it runs and returns a float
        self.assertIsInstance(kelly_small, float)

    def test_calculate_sharpe_ratio(self):
        returns = [0.01, 0.02, -0.01, 0.005]
        sharpe = self.risk_manager.calculate_sharpe_ratio(returns)
        self.assertIsInstance(sharpe, float)

        # Empty returns
        self.assertEqual(self.risk_manager.calculate_sharpe_ratio([]), 0.0)

    def test_calculate_max_drawdown(self):
        values = [100, 110, 105, 100, 90, 120]
        # Peaks: 100, 110, 110, 110, 110, 120
        # Drawdowns: 0, 0, -5/110, -10/110, -20/110, 0
        # Max DD = -20/110 = -0.1818

        pct, amount = self.risk_manager.calculate_max_drawdown(values)
        self.assertAlmostEqual(pct, (90 - 110) / 110)
        self.assertEqual(amount, 20)

    def test_assess_position_risk(self):
        risk = self.risk_manager.assess_position_risk(
            symbol="AAPL",
            position_value=1000,
            portfolio_value=100_000,
            unrealized_plpc=-0.15,  # High loss
            days_held=40,  # Long hold
        )

        # Score:
        # Pct: 1% (<= 5%) -> 0
        # PnL: -15% -> +3
        # Held: >30d & loss -> +1
        # Total: 4 -> CRITIQUE

        self.assertEqual(risk["risk_level"], "CRITIQUE")
        self.assertEqual(risk["risk_score"], 4)

    def test_get_risk_report(self):
        self.risk_manager.reset_daily_stats(100_000)
        positions = [
            {
                "symbol": "AAPL",
                "market_value": 5000,
                "unrealized_plpc": 0.02,
                "purchase_date": datetime.now(timezone.utc).isoformat(),
            }
        ]
        report = self.risk_manager.get_risk_report(positions, 100_000)
        self.assertIsInstance(report, dict)
        self.assertEqual(report["portfolio_value"], 100_000)
        self.assertEqual(report["positions_count"], 1)

    def test_log_trade(self):
        self.risk_manager.log_trade("AAPL", "BUY", 100.0)
        self.risk_manager.log_trade("AAPL", "SELL", -50.0)

        self.assertEqual(self.risk_manager.daily_trades, 2)
        self.assertEqual(self.risk_manager.daily_wins, 1)
        self.assertEqual(self.risk_manager.daily_losses, 1)
