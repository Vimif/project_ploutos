import pytest
import datetime
from unittest.mock import patch, MagicMock
from core.risk_manager import RiskManager

class TestRiskManagerCoverage:
    @pytest.fixture
    def risk_manager(self):
        return RiskManager(
            max_portfolio_risk=0.02,
            max_daily_loss=0.03,
            max_position_size=0.05,
            max_correlation=0.7
        )

    def test_initialization(self, risk_manager):
        assert risk_manager.max_portfolio_risk == 0.02
        assert risk_manager.max_daily_loss == 0.03
        assert risk_manager.max_position_size == 0.05
        assert risk_manager.daily_trades == 0
        assert risk_manager.circuit_breaker_triggered is False

    def test_calculate_position_size_normal(self, risk_manager):
        # Portfolio $100k, Entry $100, Stop Loss 5% ($95)
        # Risk Amount = $100k * 2% = $2000
        # Stop Distance = $100 * 5% = $5
        # Qty = $2000 / $5 = 400
        # Value = 400 * $100 = $40,000 (40% of portfolio) -> Should be capped at 5% ($5000)

        # Test uncapped first (increase max size to 1.0)
        risk_manager.max_position_size = 1.0
        qty, value = risk_manager.calculate_position_size(100_000, 100, 0.05)
        assert qty == 400
        assert value == 40_000

    def test_calculate_position_size_capped(self, risk_manager):
        # Default max size 5% ($5000)
        # From previous calc, $40,000 > $5,000, so it should cap
        # Capped Value = $5000
        # Qty = $5000 / $100 = 50

        qty, value = risk_manager.calculate_position_size(100_000, 100, 0.05)
        assert qty == 50
        assert value == 5_000

    def test_check_daily_loss_limit(self, risk_manager):
        # Start day at $100k
        assert risk_manager.check_daily_loss_limit(100_000) is True
        assert risk_manager.daily_start_value == 100_000

        # Loss of 2% ($98k) -> OK
        assert risk_manager.check_daily_loss_limit(98_000) is True

        # Loss of 3.1% ($96,900) -> Trigger (max 3%)
        assert risk_manager.check_daily_loss_limit(96_900) is False
        assert risk_manager.circuit_breaker_triggered is True

        # Subsequent check should still return False
        assert risk_manager.check_daily_loss_limit(100_000) is False

    def test_reset_daily_stats(self, risk_manager):
        risk_manager.daily_trades = 10
        risk_manager.circuit_breaker_triggered = True
        risk_manager.reset_daily_stats(105_000)

        assert risk_manager.daily_start_value == 105_000
        assert risk_manager.daily_trades == 0
        assert risk_manager.circuit_breaker_triggered is False

    def test_calculate_portfolio_exposure(self, risk_manager):
        positions = [
            {'market_value': 10_000},
            {'market_value': 20_000}
        ]
        exposure = risk_manager.calculate_portfolio_exposure(positions, 100_000)
        assert exposure == 0.3  # 30%

    def test_should_reduce_exposure(self, risk_manager):
        # Case 1: High exposure
        positions = [{'market_value': 90_000, 'unrealized_plpc': 0}]
        reduce, reason = risk_manager.should_reduce_exposure(positions, 100_000)
        assert reduce is True
        assert "Exposition élevée" in reason

        # Case 2: Too many losers
        positions = [
            {'market_value': 10_000, 'unrealized_plpc': -0.06},
            {'market_value': 10_000, 'unrealized_plpc': -0.06}, # 2/3 losing > 5%
            {'market_value': 10_000, 'unrealized_plpc': 0.10}
        ]
        reduce, reason = risk_manager.should_reduce_exposure(positions, 100_000)
        assert reduce is True
        assert "positions en perte" in reason

    def test_calculate_kelly_criterion(self, risk_manager):
        # Win rate 50%, Win/Loss 2.0
        # Kelly = 0.5 - (0.5 / 2.0) = 0.5 - 0.25 = 0.25 (25%)
        # Half Kelly = 12.5%
        # Capped at max_portfolio_risk (2%)

        kelly = risk_manager.calculate_kelly_criterion(0.5, 0.02, 0.01)
        assert kelly == 0.02 # Capped

        # Test negative case
        kelly = risk_manager.calculate_kelly_criterion(0.1, 0.01, 0.02)
        assert kelly == 0.0 # Should be 0 if negative expectation?
        # Kelly = 0.1 - (0.9 / 0.5) = 0.1 - 1.8 = -1.7 -> max(0, ...)

    def test_calculate_sharpe_ratio(self, risk_manager):
        returns = [0.01, 0.02, -0.01, 0.005]
        sharpe = risk_manager.calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float)
        assert risk_manager.calculate_sharpe_ratio([]) == 0.0

    def test_calculate_max_drawdown(self, risk_manager):
        values = [100, 110, 105, 100, 90, 95, 120]
        # Peak 110 -> Valley 90. Drop 20. % = 20/110 = ~18.18%
        pct, amount = risk_manager.calculate_max_drawdown(values)
        assert amount == 20
        assert abs(pct - (-0.1818)) < 0.001

    def test_assess_position_risk(self, risk_manager):
        # Safe position
        res = risk_manager.assess_position_risk('AAPL', 1000, 100_000, 0.01, 5)
        assert res['risk_level'] == 'FAIBLE'

        # Risky position (too large)
        res = risk_manager.assess_position_risk('AAPL', 10_000, 100_000, 0.01, 5)
        assert "Position trop grande" in res['warnings'][0]

        # Risky position (big loss)
        res = risk_manager.assess_position_risk('AAPL', 1000, 100_000, -0.15, 5)
        assert "Perte importante" in res['warnings'][0]

    def test_log_trade(self, risk_manager):
        risk_manager.log_trade('AAPL', 'BUY', 100)
        risk_manager.log_trade('AAPL', 'SELL', -50)
        assert risk_manager.daily_trades == 2
        assert risk_manager.daily_wins == 1
        assert risk_manager.daily_losses == 1

    def test_get_risk_report(self, risk_manager):
        positions = [{
            'symbol': 'AAPL',
            'market_value': 5000,
            'unrealized_plpc': 0.05,
            'purchase_date': datetime.datetime.now().isoformat()
        }]
        risk_manager.daily_start_value = 100_000
        report = risk_manager.get_risk_report(positions, 105_000)

        assert report['portfolio_value'] == 105_000
        assert report['positions_count'] == 1
        assert report['daily_pl'] == 5000
