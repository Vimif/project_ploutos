from datetime import datetime

import pytest

from core.risk_manager import RiskManager


class TestRiskManager:
    @pytest.fixture
    def risk_manager(self):
        return RiskManager(max_portfolio_risk=0.02, max_daily_loss=0.03, max_position_size=0.05)

    def test_initialization(self, risk_manager):
        assert risk_manager.max_portfolio_risk == 0.02
        assert risk_manager.max_daily_loss == 0.03
        assert risk_manager.max_position_size == 0.05
        assert risk_manager.daily_trades == 0
        assert not risk_manager.circuit_breaker_triggered

    def test_calculate_position_size_nominal(self, risk_manager):
        # Portfolio: 100k
        # Risk: 2% (2000$)
        # Entry: 100$
        # Stop Loss: 5% (5$)
        # Distance SL: 5$
        # Qty = 2000 / 5 = 400
        # Position Value = 400 * 100 = 40k
        # Position Size Limit: 5% of 100k = 5000$
        # So it should be capped.

        qty, val = risk_manager.calculate_position_size(
            portfolio_value=100_000, entry_price=100, stop_loss_pct=0.05
        )

        # Max position value = 5000
        # Max qty = 5000 / 100 = 50
        assert qty == 50
        assert val == 5000.0

    def test_calculate_position_size_within_limits(self, risk_manager):
        # Increase max position size to allow testing the risk formula
        risk_manager.max_position_size = 0.5  # 50%

        # Portfolio: 100k
        # Risk: 1% (1000$)
        # Entry: 100$
        # Stop: 5% (5$)
        # Qty = 1000 / 5 = 200
        # Val = 20_000 (20% of portfolio, ok)

        qty, val = risk_manager.calculate_position_size(
            portfolio_value=100_000, entry_price=100, stop_loss_pct=0.05, risk_pct=0.01
        )

        assert qty == 200
        assert val == 20000.0

    def test_check_daily_loss_limit(self, risk_manager):
        # Start day at 100k
        assert risk_manager.check_daily_loss_limit(100_000)
        assert risk_manager.daily_start_value == 100_000

        # Drop to 98k (-2%) -> OK
        assert risk_manager.check_daily_loss_limit(98_000)
        assert not risk_manager.circuit_breaker_triggered

        # Drop to 96k (-4%) -> Trigger (limit is 3%)
        assert not risk_manager.check_daily_loss_limit(96_000)
        assert risk_manager.circuit_breaker_triggered

    def test_reset_daily_stats(self, risk_manager):
        risk_manager.daily_start_value = 1000
        risk_manager.daily_trades = 5
        risk_manager.circuit_breaker_triggered = True

        risk_manager.reset_daily_stats(2000)

        assert risk_manager.daily_start_value == 2000
        assert risk_manager.daily_trades == 0
        assert not risk_manager.circuit_breaker_triggered

    def test_calculate_portfolio_exposure(self, risk_manager):
        positions = [{"market_value": 1000}, {"market_value": 2000}]
        exposure = risk_manager.calculate_portfolio_exposure(positions, 10000)
        assert exposure == 0.3  # 3000 / 10000

    def test_should_reduce_exposure(self, risk_manager):
        # Case 1: High exposure
        positions = [{"market_value": 9000, "unrealized_plpc": 0}]
        should_reduce, reason = risk_manager.should_reduce_exposure(positions, 10000)
        assert should_reduce
        assert "Exposition élevée" in reason

        # Case 2: Many losing positions
        positions = [
            {"market_value": 100, "unrealized_plpc": -0.10},
            {"market_value": 100, "unrealized_plpc": -0.10},
            {"market_value": 100, "unrealized_plpc": 0.10},
        ]
        # 2/3 losing > 5% -> 66% -> Trigger (threshold 60%)
        should_reduce, reason = risk_manager.should_reduce_exposure(positions, 10000)
        assert should_reduce
        assert "positions en perte" in reason

    def test_calculate_kelly_criterion(self, risk_manager):
        # Win rate 50%, Win/Loss ratio 2 (avg win 2%, avg loss 1%)
        # Kelly = 0.5 - (0.5 / 2) = 0.5 - 0.25 = 0.25 (25%)
        # Half Kelly = 12.5%
        # Capped at max_portfolio_risk (2%)

        kelly = risk_manager.calculate_kelly_criterion(0.5, 0.02, 0.01)
        assert kelly == 0.02  # Capped

        # Test with limits removed
        risk_manager.max_portfolio_risk = 1.0
        kelly = risk_manager.calculate_kelly_criterion(0.5, 0.02, 0.01)
        assert kelly == 0.125  # Half kelly

    def test_calculate_sharpe_ratio(self, risk_manager):
        returns = [0.01, 0.02, -0.01, 0.03, 0.01]
        sharpe = risk_manager.calculate_sharpe_ratio(returns)
        assert sharpe > 0

        # Empty returns
        assert risk_manager.calculate_sharpe_ratio([]) == 0.0

    def test_calculate_max_drawdown(self, risk_manager):
        values = [100, 110, 120, 108, 125, 110]
        # Peak 120 -> 108 (-10%)
        # Peak 125 -> 110 (-12%)

        pct, amount = risk_manager.calculate_max_drawdown(values)

        # Expected max drawdown: 125 -> 110 is 15 drop. 15/125 = 0.12 (12%)
        assert pct == pytest.approx(-0.12)
        assert amount == 15

    def test_assess_position_risk(self, risk_manager):
        # Low risk
        res = risk_manager.assess_position_risk("AAPL", 1000, 100000, 0.02, 5)
        assert res["risk_level"] == "FAIBLE"
        assert res["risk_score"] == 0

        # High risk (Big position + Big loss)
        # Pos size > 5% (max_position_size default)
        # Loss < -10%
        res = risk_manager.assess_position_risk("AAPL", 10000, 100000, -0.15, 10)
        # Score: Size(2) + Loss(3) = 5 -> CRITIQUE
        assert res["risk_level"] == "CRITIQUE"
        assert res["recommendation"] == "FERMER IMMÉDIATEMENT"

    def test_log_trade(self, risk_manager):
        risk_manager.log_trade("AAPL", "BUY")
        assert risk_manager.daily_trades == 1

        risk_manager.log_trade("AAPL", "SELL", pl=100)
        assert risk_manager.daily_trades == 2
        assert risk_manager.daily_wins == 1

        risk_manager.log_trade("AAPL", "SELL", pl=-50)
        assert risk_manager.daily_trades == 3
        assert risk_manager.daily_losses == 1

    def test_get_risk_report(self, risk_manager):
        positions = [
            {
                "symbol": "AAPL",
                "market_value": 10000,
                "unrealized_plpc": -0.15,
                "created_at": datetime.now(),
            }
        ]
        risk_manager.daily_start_value = 100000

        report = risk_manager.get_risk_report(positions, 95000)

        assert report["positions_count"] == 1
        assert report["risky_positions_count"] == 1
        assert report["daily_pl"] == -5000
        assert report["daily_pl_pct"] == -0.05
        # Since we called check_daily_loss_limit implicitly or not?
        # get_risk_report does NOT call check_daily_loss_limit, it just reports status.
        # But wait, it accesses self.circuit_breaker_triggered.

        assert not report["circuit_breaker"]
