# ruff: noqa: E402
import sys
from unittest.mock import MagicMock

if "numpy" not in sys.modules:
    sys.modules["numpy"] = MagicMock()
if "pandas" not in sys.modules:
    sys.modules["pandas"] = MagicMock()
from core.risk_manager import RiskManager


def test_risk_manager_init():
    rm = RiskManager(max_portfolio_risk=0.03, max_sector_exposure=0.35, max_correlation=0.75)
    assert rm.max_portfolio_risk == 0.03
    assert rm.max_sector_exposure == 0.35


def test_risk_manager_check_trade_basic():
    rm = RiskManager()

    # Portfolio minimal
    portfolio = {"AAPL": 10}
    current_prices = {"AAPL": 150.0, "MSFT": 300.0}

    allowed, message = rm.check_trade(
        ticker="MSFT",
        action="BUY",
        qty=5,
        price=300.0,
        portfolio=portfolio,
        current_prices=current_prices,
        equity=10000.0,
        sector="Information Technology",
    )

    assert allowed is True
    assert message == "Trade OK"


def test_risk_manager_max_sector_exposure():
    rm = RiskManager(max_sector_exposure=0.10)  # 10% max

    portfolio = {"AAPL": 100}  # $15,000 -> 15% of $100k
    current_prices = {"AAPL": 150.0, "MSFT": 300.0}

    rm.sector_map = {"AAPL": "Information Technology", "MSFT": "Information Technology"}

    allowed, message = rm.check_trade(
        ticker="MSFT",
        action="BUY",
        qty=10,
        price=300.0,
        portfolio=portfolio,
        current_prices=current_prices,
        equity=100000.0,
        sector="Information Technology",
    )

    assert allowed is False
    assert "sur-exposition au secteur" in message
