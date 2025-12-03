# config/tickers.py
"""Configuration des secteurs et tickers"""

# === FIX PATH ===
import sys
from pathlib import Path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))
# ================

SECTORS = {
    'crypto': {
        'tickers': ['BTC-USD', 'ETH-USD', 'COIN'],
        'allocation': 0.15,
        'model_name': 'brain_crypto'
    },
    'defensive': {
        'tickers': ['SPY', 'QQQ', 'VOO'],
        'allocation': 0.40,
        'model_name': 'brain_defensive'
    },
    'energy': {
        'tickers': ['XOM', 'CVX', 'XLE'],
        'allocation': 0.20,
        'model_name': 'brain_energy'
    },
    'tech': {
        'tickers': ['NVDA', 'MSFT', 'AAPL', 'GOOGL'],
        'allocation': 0.25,
        'model_name': 'brain_tech'
    }
}

# Liste plate de tous les tickers
ALL_TICKERS = []
for sector_data in SECTORS.values():
    ALL_TICKERS.extend(sector_data['tickers'])

# Mapping ticker -> secteur
TICKER_TO_SECTOR = {}
for sector, data in SECTORS.items():
    for ticker in data['tickers']:
        TICKER_TO_SECTOR[ticker] = sector
