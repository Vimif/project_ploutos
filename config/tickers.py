# config/tickers.py

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

ALL_TICKERS = []
for sector_data in SECTORS.values():
    ALL_TICKERS.extend(sector_data['tickers'])

TICKER_TO_SECTOR = {}
for sector, data in SECTORS.items():
    for ticker in data['tickers']:
        TICKER_TO_SECTOR[ticker] = sector