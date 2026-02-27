
import pytest
from unittest.mock import MagicMock
from core.risk_manager import RiskManager

class TestRiskManagerCoverage:
    def test_init(self):
        """Test initialization of RiskManager."""
        rm = RiskManager(
            max_portfolio_risk=0.01,
            max_daily_loss=0.02,
            max_position_size=0.1,
            max_correlation=0.5
        )
        assert rm.max_portfolio_risk == 0.01
        assert rm.max_daily_loss == 0.02
        assert rm.max_position_size == 0.1
        assert rm.max_correlation == 0.5
        assert rm.daily_trades == 0
        assert rm.circuit_breaker_triggered is False

    def test_calculate_position_size_basic(self):
        """Test basic position sizing."""
        # Set max_position_size to 1.0 (100%) to test pure formula without capping
        rm = RiskManager(max_portfolio_risk=0.02, max_position_size=1.0)
        portfolio_value = 100_000
        entry_price = 100
        stop_loss_pct = 0.05  # 5% stop loss

        # Risk amount = 100,000 * 0.02 = 2,000
        # Stop distance = 100 * 0.05 = 5
        # Quantity = 2,000 / 5 = 400

        qty, value = rm.calculate_position_size(portfolio_value, entry_price, stop_loss_pct)

        assert qty == 400
        assert value == 400 * 100

    def test_calculate_position_size_capped(self):
        """Test position sizing capped by max position size."""
        # Set max position size small to force capping
        rm = RiskManager(max_portfolio_risk=0.05, max_position_size=0.1)
        portfolio_value = 100_000
        entry_price = 100
        stop_loss_pct = 0.01 # 1% stop, very tight

        # Risk amount = 5,000
        # Stop distance = 1
        # Raw Quantity = 5,000. Value = 500,000.
        # Max position value = 100,000 * 0.1 = 10,000.
        # Expected quantity = 10,000 / 100 = 100.

        qty, value = rm.calculate_position_size(portfolio_value, entry_price, stop_loss_pct)

        assert qty == 100
        assert value == 10_000

    def test_check_daily_loss_limit(self):
        """Test circuit breaker logic."""
        rm = RiskManager(max_daily_loss=0.05)

        # Day start
        assert rm.check_daily_loss_limit(100_000) is True
        assert rm.daily_start_value == 100_000

        # Small loss
        assert rm.check_daily_loss_limit(98_000) is True

        # Big loss (>5%)
        # 100k -> 94k is -6k, which is -6%. Limit is 5%.
        # check_daily_loss_limit returns False if limit breached.
        assert rm.check_daily_loss_limit(94_000) is False
        assert rm.circuit_breaker_triggered is True

        # Recovery to 96k (-4%)
        # Logic in RiskManager:
        # if daily_pl_pct <= -self.max_daily_loss: return False
        # It does NOT check self.circuit_breaker_triggered at the start.
        # It re-evaluates strictly based on current P&L vs start value.
        # -4% > -5%, so it returns True (allowed), effectively resetting the trading capability
        # even if the flag self.circuit_breaker_triggered remains True (for logging/reporting).

        # Based on code analysis, it returns True here.
        assert rm.check_daily_loss_limit(96_000) is True

    def test_reset_daily_stats(self):
        """Test resetting daily stats."""
        rm = RiskManager()
        rm.daily_trades = 10
        rm.circuit_breaker_triggered = True

        rm.reset_daily_stats(105_000)

        assert rm.daily_trades == 0
        assert rm.circuit_breaker_triggered is False
        assert rm.daily_start_value == 105_000

    def test_calculate_portfolio_exposure(self):
        """Test exposure calculation."""
        rm = RiskManager()
        positions = [
            {'market_value': 10_000},
            {'market_value': 20_000}
        ]
        portfolio_value = 100_000

        exposure = rm.calculate_portfolio_exposure(positions, portfolio_value)
        assert exposure == 0.3

    def test_should_reduce_exposure(self):
        """Test exposure reduction logic."""
        rm = RiskManager()

        # Case 1: High exposure
        positions_high = [{'market_value': 90_000, 'unrealized_plpc': 0.0}]
        should_reduce, reason = rm.should_reduce_exposure(positions_high, 100_000)
        assert should_reduce is True
        assert "Exposition élevée" in reason

        # Case 2: Many losing positions
        positions_losing = [
            {'market_value': 1000, 'unrealized_plpc': -0.06},
            {'market_value': 1000, 'unrealized_plpc': -0.06},
            {'market_value': 1000, 'unrealized_plpc': 0.01}
        ]
        # 2/3 losing > 5% -> 66% which is > 60%
        should_reduce, reason = rm.should_reduce_exposure(positions_losing, 100_000)
        assert should_reduce is True
        assert "positions en perte" in reason

    def test_calculate_kelly_criterion(self):
        """Test Kelly criterion."""
        rm = RiskManager(max_portfolio_risk=0.1)

        # Win rate 50%, Win/Loss ratio 2.0
        # Kelly = 0.5 - (0.5 / 2.0) = 0.5 - 0.25 = 0.25
        # Half Kelly = 0.125
        # Capped at max_risk 0.1

        kelly = rm.calculate_kelly_criterion(0.5, 0.2, 0.1)
        assert kelly == 0.1

        # Zero cases
        assert rm.calculate_kelly_criterion(0, 0.1, 0.1) == 0.1 # Fallback to max risk? Logic check.
        # Code: if avg_loss == 0 or win_rate == 0: return self.max_portfolio_risk

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe calculation."""
        rm = RiskManager()
        returns = [0.01, 0.02, -0.01, 0.01]
        sharpe = rm.calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float)

    def test_calculate_max_drawdown(self):
        """Test Max Drawdown."""
        rm = RiskManager()
        values = [100, 110, 105, 100, 120]
        # Peak 110 -> Valley 100. DD = (100-110)/110 = -9.09%
        # Peak 120 -> no subsequent valley yet.

        dd_pct, dd_amt = rm.calculate_max_drawdown(values)
        assert dd_pct < 0
        assert dd_amt > 0

    def test_assess_position_risk(self):
        """Test position risk assessment."""
        rm = RiskManager(max_position_size=0.1)

        # Risky position (too big, big loss)
        risk = rm.assess_position_risk(
            symbol="TEST",
            position_value=20_000, # 20% > 10%
            portfolio_value=100_000,
            unrealized_plpc=-0.15, # -15%
            days_held=5
        )

        assert risk['risk_score'] >= 4
        assert risk['risk_level'] == "CRITIQUE"

    def test_log_trade(self):
        rm = RiskManager()
        rm.log_trade("TEST", "BUY", 100)
        assert rm.daily_wins == 1
        assert rm.daily_trades == 1

        rm.log_trade("TEST", "SELL", -50)
        assert rm.daily_losses == 1
        assert rm.daily_trades == 2

    def test_get_risk_report(self):
        rm = RiskManager()
        report = rm.get_risk_report([], 100_000)
        assert 'exposure_pct' in report
        assert 'portfolio_value' in report
