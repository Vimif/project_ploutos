import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# Mock numpy before it's imported by core.risk_manager
mock_np = MagicMock()
sys.modules["numpy"] = mock_np

# Mock core.utils to avoid import errors
mock_utils = MagicMock()
sys.modules["core.utils"] = mock_utils
# Ensure setup_logging returns a mock logger
mock_logger = MagicMock()
mock_utils.setup_logging.return_value = mock_logger

# Ajouter la racine du projet au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.risk_manager import RiskManager

def test_days_held_calculation():
    rm = RiskManager()

    # 1. Position avec purchase_date d'il y a 40 jours
    date_40_days_ago = (datetime.now() - timedelta(days=40)).isoformat()
    # 2. Position avec created_at (fallback Alpaca) d'il y a 45 jours
    date_45_days_ago = (datetime.now() - timedelta(days=45)).isoformat()

    positions = [
        {
            'symbol': 'AAPL',
            'market_value': 5000,
            'unrealized_plpc': -0.06,
            'purchase_date': date_40_days_ago
        },
        {
            'symbol': 'NVDA',
            'market_value': 6000,
            'unrealized_plpc': -0.07,
            'created_at': date_45_days_ago # Test fallback Alpaca
        }
    ]

    portfolio_value = 100000
    report = rm.get_risk_report(positions, portfolio_value)

    print(f"Positions à risque: {report['risky_positions_count']}")

    found_aapl = False
    found_nvda = False
    for pos in report['risky_positions']:
        print(f"Position à risque: {pos['symbol']}, Score: {pos['risk_score']}, Level: {pos['risk_level']}")
        if pos['symbol'] == 'AAPL':
            found_aapl = True
            assert pos['risk_score'] >= 2
            assert any("Perte prolongée" in w for w in pos['warnings'])
            assert any("40 jours" in w for w in pos['warnings'])
        if pos['symbol'] == 'NVDA':
            found_nvda = True
            assert pos['risk_score'] >= 2
            assert any("Perte prolongée" in w for w in pos['warnings'])
            assert any("45 jours" in w for w in pos['warnings'])

    assert found_aapl, "AAPL should be risky"
    assert found_nvda, "NVDA should be risky (fallback created_at)"
    print("✅ Tests de calcul days_held réussis !")

if __name__ == "__main__":
    test_days_held_calculation()
