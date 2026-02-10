"""Configuration des secteurs et tickers - S&P 500 GICS dynamique"""

import sys
import json
from pathlib import Path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

# ======================================================================
# 11 secteurs GICS (S&P 500 standard)
# Les tickers sont remplis dynamiquement via le scanner
# ======================================================================
GICS_SECTORS = {
    'information_technology': {
        'name': 'Information Technology',
        'tickers': [],
        'allocation': 0.15,
    },
    'health_care': {
        'name': 'Health Care',
        'tickers': [],
        'allocation': 0.12,
    },
    'financials': {
        'name': 'Financials',
        'tickers': [],
        'allocation': 0.11,
    },
    'consumer_discretionary': {
        'name': 'Consumer Discretionary',
        'tickers': [],
        'allocation': 0.10,
    },
    'communication_services': {
        'name': 'Communication Services',
        'tickers': [],
        'allocation': 0.09,
    },
    'industrials': {
        'name': 'Industrials',
        'tickers': [],
        'allocation': 0.09,
    },
    'consumer_staples': {
        'name': 'Consumer Staples',
        'tickers': [],
        'allocation': 0.08,
    },
    'energy': {
        'name': 'Energy',
        'tickers': [],
        'allocation': 0.09,
    },
    'utilities': {
        'name': 'Utilities',
        'tickers': [],
        'allocation': 0.06,
    },
    'real_estate': {
        'name': 'Real Estate',
        'tickers': [],
        'allocation': 0.06,
    },
    'materials': {
        'name': 'Materials',
        'tickers': [],
        'allocation': 0.05,
    },
}

# Mapping nom GICS Wikipedia -> cle interne
_SECTOR_NAME_MAP = {
    'Information Technology': 'information_technology',
    'Health Care': 'health_care',
    'Financials': 'financials',
    'Consumer Discretionary': 'consumer_discretionary',
    'Communication Services': 'communication_services',
    'Industrials': 'industrials',
    'Consumer Staples': 'consumer_staples',
    'Energy': 'energy',
    'Utilities': 'utilities',
    'Real Estate': 'real_estate',
    'Materials': 'materials',
}

# ======================================================================
# Legacy (V6) — compatibilite arriere
# ======================================================================
SECTORS_LEGACY = {
    'growth': {
        'tickers': ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN'],
        'allocation': 0.30,
        'model_name': 'brain_growth',
    },
    'defensive': {
        'tickers': ['SPY', 'QQQ', 'VOO', 'VTI'],
        'allocation': 0.40,
        'model_name': 'brain_defensive',
    },
    'energy': {
        'tickers': ['XOM', 'CVX', 'COP', 'XLE'],
        'allocation': 0.15,
        'model_name': 'brain_energy',
    },
    'finance': {
        'tickers': ['JPM', 'BAC', 'WFC', 'GS'],
        'allocation': 0.15,
        'model_name': 'brain_finance',
    },
}


def get_legacy_tickers():
    """Retourne les 15 tickers hardcodes V6."""
    return [
        'NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'TSLA',
        'SPY', 'QQQ', 'VOO', 'VTI', 'XLE', 'XLF', 'XLK', 'XLV',
    ]


# ======================================================================
# Chargement dynamique depuis les resultats du scanner
# ======================================================================
def load_dynamic_tickers(
    scan_path: str = 'data/sp500_cache/latest_scan.json',
) -> dict:
    """Charge les tickers depuis le JSON du scanner et peuple GICS_SECTORS.

    Returns:
        GICS_SECTORS avec les tickers remplis.
    """
    path = Path(scan_path)
    if not path.exists():
        print(f"[WARN] Scan results not found: {scan_path}")
        print("       Run: python core/sp500_scanner.py")
        return GICS_SECTORS

    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    for sector_name, tickers in data.get('sectors', {}).items():
        key = _SECTOR_NAME_MAP.get(sector_name)
        if key and key in GICS_SECTORS:
            GICS_SECTORS[key]['tickers'] = tickers

    total = sum(len(s['tickers']) for s in GICS_SECTORS.values())
    print(f"Loaded {total} tickers from scan ({data.get('scan_date', '?')})")
    return GICS_SECTORS


def get_all_tickers(use_dynamic: bool = True) -> list:
    """Liste plate de tous les tickers (dynamiques ou legacy)."""
    if use_dynamic:
        sectors = load_dynamic_tickers()
    else:
        return get_legacy_tickers()

    tickers = []
    for sector_data in sectors.values():
        tickers.extend(sector_data.get('tickers', []))
    return tickers


# ======================================================================
# Expose SECTORS comme alias par defaut
# ======================================================================
SECTORS = GICS_SECTORS

ALL_TICKERS = []
for _sd in SECTORS.values():
    ALL_TICKERS.extend(_sd.get('tickers', []))

TICKER_TO_SECTOR = {}
for _sect_key, _sect_data in SECTORS.items():
    for _t in _sect_data.get('tickers', []):
        TICKER_TO_SECTOR[_t] = _sect_key
