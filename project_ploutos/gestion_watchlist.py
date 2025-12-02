# gestion_watchlist.py
import json
import os

FICHIER_WL = "watchlist.json"
# Liste de base si le fichier est vide
DEFAULT_LIST = ["AAPL", "TSLA", "MSFT", "NVDA", "AMD", "INTC", "AMZN", "GOOGL", "META", "NFLX", "KO", "MCD", "DIS", "JPM", "BAC"]

class WatchlistManager:
    def __init__(self):
        self.tickers = self.charger()

    def charger(self):
        if os.path.exists(FICHIER_WL):
            try:
                with open(FICHIER_WL, 'r') as f:
                    return json.load(f)
            except: return DEFAULT_LIST
        return DEFAULT_LIST

    def sauvegarder(self):
        with open(FICHIER_WL, 'w') as f:
            json.dump(self.tickers, f)

    def ajouter(self, ticker):
        ticker = ticker.upper().strip()
        if ticker and ticker not in self.tickers:
            self.tickers.append(ticker)
            self.sauvegarder()
            return True
        return False

    def retirer(self, ticker):
        if ticker in self.tickers:
            self.tickers.remove(ticker)
            self.sauvegarder()
            return True
        return False
    
    def get_list(self):
        return self.tickers
