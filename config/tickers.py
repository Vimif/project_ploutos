"""Configuration des secteurs et tickers - ACTIONS UNIQUEMENT"""

# === FIX PATH ===
import sys
from pathlib import Path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))
# ================

SECTORS = {
    'growth': {
        # Actions tech/croissance
        'tickers': ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN'],
        'allocation': 0.30,
        'model_name': 'brain_growth'
    },
    'defensive': {
        # ETFs défensifs
        'tickers': ['SPY', 'QQQ', 'VOO', 'VTI'],
        'allocation': 0.40,
        'model_name': 'brain_defensive'
    },
    'energy': {
        # Secteur énergie
        'tickers': ['XOM', 'CVX', 'COP', 'XLE'],
        'allocation': 0.15,
        'model_name': 'brain_energy'
    },
    'finance': {
        # Secteur financier
        'tickers': ['JPM', 'BAC', 'WFC', 'GS'],
        'allocation': 0.15,
        'model_name': 'brain_finance'
    }
}

ALL_TICKERS = []
for sector_data in SECTORS.values():
    ALL_TICKERS.extend(sector_data['tickers'])

TICKER_TO_SECTOR = {}
for sector, data in SECTORS.items():
    for ticker in data['tickers']:
        TICKER_TO_SECTOR[ticker] = sector

# Stats
print(f"📊 Configuration chargée:")
print(f"   - Secteurs: {len(SECTORS)}")
print(f"   - Tickers totaux: {len(ALL_TICKERS)}")
for sector, config in SECTORS.items():
    print(f"   - {sector}: {len(config['tickers'])} tickers ({config['allocation']*100:.0f}%)")