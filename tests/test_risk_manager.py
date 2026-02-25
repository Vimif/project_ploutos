import pytest

from core.risk_manager import RiskManager


class TestRiskManager:
    @pytest.fixture
    def risk_manager(self):
        return RiskManager(max_portfolio_risk=0.02, max_daily_loss=0.03, max_position_size=1.0)

    def test_initialization(self, risk_manager):
        assert risk_manager.max_portfolio_risk == 0.02
        assert risk_manager.max_daily_loss == 0.03
        assert risk_manager.max_position_size == 1.0
        assert risk_manager.daily_trades == 0
        assert not risk_manager.circuit_breaker_triggered

    def test_calculate_position_size(self, risk_manager):
        portfolio_value = 100_000
        entry_price = 150.0
        stop_loss_pct = 0.05  # 5% stop loss

        # Risk amount = 100k * 2% = 2000
        # Stop distance = 150 * 5% = 7.5
        # Qty = 2000 / 7.5 = 266.66 -> 266

        qty, value = risk_manager.calculate_position_size(
            portfolio_value, entry_price, stop_loss_pct
        )

        assert qty == 266
        assert value == 266 * 150.0

        risk_manager.max_position_size = 0.05
        # Check max position size constraint
        # Max pos = 100k * 5% = 5000
        # If we set very tight stop loss, qty increases
        stop_loss_pct_tight = 0.01  # 1.5 stop distance
        # Risk amount = 2000
        # Qty = 2000 / 1.5 = 1333
        # Value = 1333 * 150 = 199,950 >> 5000

        qty_capped, value_capped = risk_manager.calculate_position_size(
            portfolio_value, entry_price, stop_loss_pct_tight
        )

        assert value_capped <= 5000 + 150  # allow rounding margin (one share)
        assert qty_capped * entry_price <= 5000 + 150

    def test_check_daily_loss_limit(self, risk_manager):
        # Day start
        risk_manager.check_daily_loss_limit(100_000)
        assert risk_manager.daily_start_value == 100_000

        # Small loss (-1%)
        assert risk_manager.check_daily_loss_limit(99_000) is True
        assert not risk_manager.circuit_breaker_triggered

        # Big loss (-4%) > 3% limit
        assert risk_manager.check_daily_loss_limit(96_000) is False
        assert risk_manager.circuit_breaker_triggered

        # Recovery (should still be triggered)
        assert risk_manager.check_daily_loss_limit(98_000) is False

    def test_reset_daily_stats(self, risk_manager):
        risk_manager.daily_start_value = 100_000
        risk_manager.circuit_breaker_triggered = True
        risk_manager.daily_trades = 10

        risk_manager.reset_daily_stats(105_000)

        assert risk_manager.daily_start_value == 105_000
        assert risk_manager.daily_trades == 0
        assert not risk_manager.circuit_breaker_triggered

    def test_calculate_portfolio_exposure(self, risk_manager):
        positions = [{"market_value": 10_000}, {"market_value": 20_000}]
        portfolio_value = 100_000

        exposure = risk_manager.calculate_portfolio_exposure(positions, portfolio_value)
        assert exposure == 0.30  # 30%

    def test_should_reduce_exposure(self, risk_manager):
        positions = [{"market_value": 90_000, "unrealized_plpc": 0.01}]  # 90% exposure
        reduce, reason = risk_manager.should_reduce_exposure(positions, 100_000)
        assert reduce is True
        assert "Exposition élevée" in reason

        positions_losing = [
            {"market_value": 1000, "unrealized_plpc": -0.10},
            {"market_value": 1000, "unrealized_plpc": -0.06},
            {"market_value": 1000, "unrealized_plpc": 0.01},
        ]
        # 2/3 losing > 5%
        reduce, reason = risk_manager.should_reduce_exposure(positions_losing, 100_000)
        assert reduce is True
        assert "positions en perte" in reason

    def test_kelly_criterion(self, risk_manager):
        # W=0.6, R=2.0 (Avg win 2%, Avg loss 1%)
        # Kelly = 0.6 - (0.4 / 2) = 0.6 - 0.2 = 0.4 (40%)
        # Half Kelly = 20%
        # Capped at max_portfolio_risk (2%)

        kelly = risk_manager.calculate_kelly_criterion(0.6, 0.02, 0.01)
        assert kelly == 0.02

        # Test uncapped (if max risk was high)
        risk_manager.max_portfolio_risk = 1.0
        kelly = risk_manager.calculate_kelly_criterion(0.6, 0.02, 0.01)
        assert kelly == pytest.approx(0.20)  # Half kelly

    def test_sharpe_ratio(self, risk_manager):
        returns = [0.01, 0.02, -0.01, 0.005]
        sharpe = risk_manager.calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float)

        # Empty returns
        assert risk_manager.calculate_sharpe_ratio([]) == 0.0

    def test_max_drawdown(self, risk_manager):
        # 100 -> 110 -> 99 (-10% from peak) -> 120
        values = [100, 110, 99, 120]
        pct, amount = risk_manager.calculate_max_drawdown(values)

        # Peak 110, Low 99. Drop 11. 11/110 = 0.10
        assert pct == pytest.approx(-0.10)
        assert amount == 11

    def test_assess_position_risk(self, risk_manager):
        risk_manager.max_position_size = 0.05
        res = risk_manager.assess_position_risk(
            symbol="TEST",
            position_value=6000,  # 6% > 5% max
            portfolio_value=100_000,
            unrealized_plpc=-0.12,  # -12% < -10%
            days_held=40,
        )

        # Score:
        # Size > 5%: +2
        # Loss < -10%: +3
        # Held > 30d & loss: +1
        # Total: 6 -> CRITIQUE

        assert res["risk_level"] == "CRITIQUE"
        assert res["risk_score"] == 6
