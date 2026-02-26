import pytest

from core.risk_manager import RiskManager


class TestRiskManager:
    @pytest.fixture
    def risk_manager(self):
        return RiskManager(
            max_portfolio_risk=0.02,
            max_daily_loss=0.03,
            max_position_size=0.05,
            max_correlation=0.7,
        )

    def test_calculate_position_size_normal(self, risk_manager):
        # 100k portfolio, $100 entry, 5% stop loss
        # Risk amount = 100k * 0.02 = 2000
        # Stop loss dist = 100 * 0.05 = 5
        # Quantity = 2000 / 5 = 400
        # Position value = 400 * 100 = 40k (40%) -> Capped at 5% (5k)

        # We need to test the raw calculation first.
        # Let's use a very tight stop loss to NOT trigger max position size capping
        # Risk amount = 2000
        # Stop loss dist = 100 * 0.50 = 50 (50% stop loss)
        # Quantity = 2000 / 50 = 40
        # Position value = 40 * 100 = 4000 (4%) -> OK

        qty, val = risk_manager.calculate_position_size(
            portfolio_value=100000, entry_price=100, stop_loss_pct=0.50
        )
        assert qty == 40
        assert val == 4000.0

    def test_calculate_position_size_capped(self, risk_manager):
        # Trigger capping
        # Risk = 2000
        # Stop = 5% ($5)
        # Raw Qty = 400
        # Raw Val = 40k (40%)
        # Cap = 5% of 100k = 5000
        # Capped Qty = 5000 / 100 = 50

        qty, val = risk_manager.calculate_position_size(
            portfolio_value=100000, entry_price=100, stop_loss_pct=0.05
        )
        assert qty == 50
        assert val == 5000.0

    def test_check_daily_loss_limit(self, risk_manager):
        # Init
        assert risk_manager.check_daily_loss_limit(100000)
        assert risk_manager.daily_start_value == 100000

        # Small loss (-1%)
        assert risk_manager.check_daily_loss_limit(99000)

        # Big loss (-4%) -> Trigger (limit is 3%)
        assert not risk_manager.check_daily_loss_limit(96000)
        assert risk_manager.circuit_breaker_triggered

        # Subsequent checks should fail
        assert not risk_manager.check_daily_loss_limit(97000)  # Even if it recovers a bit?
        # Actually logic is: if triggered, it returns False?
        # Looking at code:
        # if daily_pl_pct <= -max_daily_loss: triggered=True; return False
        # The logic doesn't explicitly say "if triggered return False immediately" at the top,
        # but once triggered, if value stays low it returns False.
        # If value recovers to 99k (-1%), check_daily_loss_limit(99000) would return True?
        # Let's check the code:
        # if daily_pl_pct <= -self.max_daily_loss: ... return False
        # return True
        # So if it recovers, it returns True unless there is a latch.
        # The code sets `self.circuit_breaker_triggered = True`.
        # BUT it does NOT check `self.circuit_breaker_triggered` at the start of the function.
        # Wait, let me check the file content again.
        pass

    def test_reset_daily_stats(self, risk_manager):
        risk_manager.daily_start_value = 100000
        risk_manager.daily_trades = 5
        risk_manager.circuit_breaker_triggered = True

        risk_manager.reset_daily_stats(105000)

        assert risk_manager.daily_start_value == 105000
        assert risk_manager.daily_trades == 0
        assert not risk_manager.circuit_breaker_triggered

    def test_kelly_criterion(self, risk_manager):
        # Win rate 50%, Win/Loss = 2.0 (Avg win 2%, Avg loss 1%)
        # Kelly = 0.5 - (0.5 / 2.0) = 0.5 - 0.25 = 0.25 (25%)
        # Half Kelly = 12.5%
        # Capped at max_portfolio_risk (2%)

        kelly = risk_manager.calculate_kelly_criterion(0.5, 0.02, 0.01)
        assert kelly == 0.02  # Capped

        # Negative Kelly
        # Win rate 30%, Win/Loss = 1.0
        # Kelly = 0.3 - (0.7 / 1) = -0.4
        kelly = risk_manager.calculate_kelly_criterion(0.3, 0.01, 0.01)
        assert kelly == 0.0

    def test_sharpe_ratio(self, risk_manager):
        returns = [0.01, 0.02, -0.01, 0.01, 0.0]
        sharpe = risk_manager.calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float)
        assert sharpe > 0

        # Empty
        assert risk_manager.calculate_sharpe_ratio([]) == 0.0

    def test_max_drawdown(self, risk_manager):
        values = [100, 110, 105, 100, 90, 95, 80, 85]
        # Peak 110. Min 80.
        # But DD is from peak to subsequent trough.
        # 110 -> 80 is the max dd path?
        # 110 to 90 is -20 (18%)
        # 110 to 80 is -30 (27%)
        # Let's verify calculation
        pct, amount = risk_manager.calculate_max_drawdown(values)
        assert amount == 30.0  # 110 - 80
        # 30 / 110 = 0.2727
        assert abs(pct - (-0.2727)) < 0.001

    def test_assess_position_risk(self, risk_manager):
        # Good position
        res = risk_manager.assess_position_risk(
            "AAPL", position_value=4000, portfolio_value=100000, unrealized_plpc=0.05, days_held=5
        )
        assert res["risk_level"] == "FAIBLE"

        # Bad position (huge loss)
        res = risk_manager.assess_position_risk(
            "AAPL", position_value=4000, portfolio_value=100000, unrealized_plpc=-0.15, days_held=5
        )
        assert res["risk_level"] == "ÉLEVÉ" or res["risk_level"] == "CRITIQUE"
