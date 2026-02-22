import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Need to properly mock numpy and core.utils before importing RiskManager
# However, pytest's patching mechanism usually works better than sys.modules hacking for tests
# But since the file already imports them, we might need to rely on the imports being resolvable.
# Let's try to import RiskManager directly, assuming dependencies are installed in the test env.
# If not, we fall back to mocks.

try:
    from core.risk_manager import RiskManager
except ImportError:
    # If imports fail (e.g. numpy missing in minimal test env), mock them
    mock_np = MagicMock()
    sys.modules["numpy"] = mock_np

    mock_utils = MagicMock()
    sys.modules["core.utils"] = mock_utils
    mock_utils.setup_logging.return_value = MagicMock()

    from core.risk_manager import RiskManager


class TestRiskManager:
    def test_days_held_calculation(self):
        rm = RiskManager()

        # 1. Position with purchase_date 40 days ago
        date_40_days_ago = (datetime.now() - timedelta(days=40)).isoformat()
        # 2. Position with created_at (Alpaca fallback) 45 days ago
        date_45_days_ago = (datetime.now() - timedelta(days=45)).isoformat()

        positions = [
            {
                "symbol": "AAPL",
                "market_value": 5000,
                "unrealized_plpc": -0.06,
                "purchase_date": date_40_days_ago,
            },
            {
                "symbol": "NVDA",
                "market_value": 6000,
                "unrealized_plpc": -0.07,
                "created_at": date_45_days_ago,  # Test fallback Alpaca
            },
        ]

        portfolio_value = 100000
        report = rm.get_risk_report(positions, portfolio_value)

        found_aapl = False
        found_nvda = False

        for pos in report["risky_positions"]:
            if pos["symbol"] == "AAPL":
                found_aapl = True
                assert pos["risk_score"] >= 2
                assert any("Perte prolongée" in w for w in pos["warnings"])
                assert any("40 jours" in w for w in pos["warnings"])
            if pos["symbol"] == "NVDA":
                found_nvda = True
                assert pos["risk_score"] >= 2
                assert any("Perte prolongée" in w for w in pos["warnings"])
                assert any("45 jours" in w for w in pos["warnings"])

        assert found_aapl, "AAPL should be identified as risky"
        assert found_nvda, "NVDA should be identified as risky (fallback created_at)"

    def test_concentration_risk(self):
        rm = RiskManager()
        # Position > 20% of portfolio
        positions = [
            {
                "symbol": "TSLA",
                "market_value": 25000,  # 25% of 100k
                "unrealized_plpc": 0.1,
                "purchase_date": datetime.now().isoformat(),
            }
        ]
        report = rm.get_risk_report(positions, 100000)

        found_tsla = False
        for pos in report["risky_positions"]:
            if pos["symbol"] == "TSLA":
                found_tsla = True
                assert any("Position trop grande" in w for w in pos["warnings"])

        assert found_tsla, "TSLA should be flagged for concentration"

    def test_calculate_position_size(self):
        rm = RiskManager()
        # Account 100k, price 100. Max pos size usually 5% or 10% depending on settings.
        # Default max_position_size in RiskManager is usually 0.1 (10%) or 0.05 (5%)
        # Let's check config or default. Assuming default is conservative.

        # Test 1: Standard calculation
        qty, val = rm.calculate_position_size(
            portfolio_value=100000, entry_price=150.0, stop_loss_pct=0.05
        )
        assert qty > 0

        # Test 2: High volatility (larger stop loss) should reduce size
        # NOTE: If max_position_size (5%) is the limiting factor, both might be equal (33 shares)
        # We check that it doesn't INCREASE.
        qty_high_vol, val_high_vol = rm.calculate_position_size(
            portfolio_value=100000, entry_price=150.0, stop_loss_pct=0.10
        )
        assert qty_high_vol <= qty, "Larger stop loss should not increase position size"

    def test_check_daily_loss_limit(self):
        rm = RiskManager()
        rm.reset_daily_stats(100000)

        # Initial check
        can_trade = rm.check_daily_loss_limit(current_value=100000)
        assert can_trade

        # Simulate loss > max_daily_loss (default usually 2-3%)
        # If equity drops to 95k (5% loss)
        can_trade = rm.check_daily_loss_limit(current_value=95000)
        assert not can_trade

        # Should stay triggered even if equity recovers slightly (if logic persists)
        can_trade = rm.check_daily_loss_limit(current_value=96000)
        assert not can_trade


if __name__ == "__main__":
    pytest.main([__file__])
