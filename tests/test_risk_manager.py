import pytest

from core.risk_manager import RiskManager


class TestRiskManager:
    @pytest.fixture
    def risk_manager(self):
        return RiskManager(
            max_portfolio_risk=0.02,
            max_daily_loss=0.03,
            max_position_size=0.20,
            max_correlation=0.7,
        )

    def test_initialization(self, risk_manager):
        assert risk_manager.max_portfolio_risk == 0.02
        assert risk_manager.max_daily_loss == 0.03
        assert risk_manager.daily_start_value is None
        assert risk_manager.circuit_breaker_triggered is False

    def test_calculate_position_size(self, risk_manager):
        # Portfolio 100k, Risk 2% (2k), Stop 5%
        # Size = 2000 / (150 * 0.05) = 2000 / 7.5 = 266.6 -> 266
        qty, val = risk_manager.calculate_position_size(
            portfolio_value=100000.0, entry_price=150.0, stop_loss_pct=0.05, risk_pct=0.02
        )
        assert qty == 133
        assert val == 133 * 150.0
        assert val <= 100000.0 * 0.20  # Check max pos size (20k)
        # 266 * 150 = 39900 > 20000
        # It should have been capped!
        # Re-calculating with cap:
        # Max pos = 20k. 20k / 150 = 133.
        # Let's check logic: calculate_position_size caps at max_position_size.
        assert qty <= 133

    def test_check_daily_loss_limit(self, risk_manager):
        # Day start
        risk_manager.reset_daily_stats(100000.0)
        assert risk_manager.daily_start_value == 100000.0

        # Small loss (-1k = -1%) -> OK (Limit 3%)
        assert risk_manager.check_daily_loss_limit(99000.0) is True
        assert risk_manager.circuit_breaker_triggered is False

        # Big loss (-4k = -4%) -> Trigger
        assert risk_manager.check_daily_loss_limit(96000.0) is False
        assert risk_manager.circuit_breaker_triggered is True

        # Should stay triggered even if recovery
        assert risk_manager.check_daily_loss_limit(98000.0) is False

    def test_reset_daily_stats(self, risk_manager):
        risk_manager.circuit_breaker_triggered = True
        risk_manager.reset_daily_stats(100000.0)
        assert risk_manager.circuit_breaker_triggered is False
        assert risk_manager.daily_start_value == 100000.0

    def test_calculate_portfolio_exposure(self, risk_manager):
        positions = [{"market_value": 10000.0}, {"market_value": 20000.0}]
        exposure = risk_manager.calculate_portfolio_exposure(positions, 100000.0)
        assert exposure == 0.30

    def test_assess_position_risk(self, risk_manager):
        risk = risk_manager.assess_position_risk(
            symbol="TEST",
            position_value=25000.0,  # 25% > 20% limit
            portfolio_value=100000.0,
            unrealized_plpc=-0.06,  # -6%
            days_held=5,
        )
        assert risk["risk_score"] >= 2  # 2 (size) + 1 (loss) = 3
        assert "Position trop grande" in str(risk["warnings"])
        assert "Perte significative" in str(risk["warnings"])

    def test_calculate_sharpe_ratio(self, risk_manager):
        returns = [0.01, -0.005, 0.01, 0.002]
        sharpe = risk_manager.calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float)
        assert sharpe > 0  # Positive mean

    def test_calculate_max_drawdown(self, risk_manager):
        values = [100, 110, 100, 90, 120]
        dd_pct, dd_amt = risk_manager.calculate_max_drawdown(values)
        # Peak 110 -> 90. Drop 20. 20/110 = 18.18%
        assert abs(dd_pct + 0.1818) < 0.01  # dd is negative usually?
        # Code says: drawdowns = (values - cumulative_max) / cumulative_max. So negative.
        # np.min(drawdowns). So -0.18.
        # Wait, implementation:
        # max_dd_pct = np.min(drawdowns)
        assert dd_pct < 0

    def test_get_risk_report(self, risk_manager):
        positions = [
            {
                "symbol": "A",
                "market_value": 10000.0,
                "unrealized_plpc": 0.05,
                "purchase_date": "2023-01-01",
            },
            {
                "symbol": "B",
                "market_value": 25000.0,
                "unrealized_plpc": -0.15,
                "created_at": "2023-01-01",
            },
        ]
        report = risk_manager.get_risk_report(positions, 100000.0, daily_returns=[0.01, -0.01])
        assert report["portfolio_value"] == 100000.0
        assert report["positions_count"] == 2
        assert len(report["risky_positions"]) >= 1  # B is too big and losing

    def test_print_risk_summary(self, risk_manager):
        report = {
            "portfolio_value": 100000.0,
            "exposure_pct": 0.5,
            "positions_count": 2,
            "risky_positions_count": 1,
            "sharpe_ratio": 1.5,
            "daily_pl": 500.0,
            "daily_pl_pct": 0.005,
            "daily_trades": 3,
            "daily_wins": 2,
            "daily_losses": 1,
            "circuit_breaker": False,
            "risky_positions": [
                {
                    "symbol": "B",
                    "risk_level": "HIGH",
                    "recommendation": "SELL",
                    "warnings": ["Too big"],
                }
            ],
        }
        # Just check it runs without error
        risk_manager.print_risk_summary(report)

    def test_log_trade(self, risk_manager):
        risk_manager.log_trade("AAPL", "buy", 100.0)
        assert risk_manager.daily_trades == 1
        assert risk_manager.daily_wins == 1
        assert risk_manager.daily_losses == 0

        risk_manager.log_trade("MSFT", "sell", -50.0)
        assert risk_manager.daily_trades == 2
        assert risk_manager.daily_wins == 1
        assert risk_manager.daily_losses == 1
