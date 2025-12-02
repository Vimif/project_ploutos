import requests
import pandas as pd
import numpy as np
import time

# --- CONFIGURATION ---
ALPHA_VANTAGE_KEY = "N7Y23TIA23OZ892L"

# LISTE DES ACTIONS A SCANNER (Modifiez selon vos envies)
# Mélange Tech US, Luxe FR, Crypto-Stocks, etc.
WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", # GAFAM
    "AMD", "INTC", "IBM", "ORCL",                    # Tech
    "KO", "PEP", "MCD",                              # Conso
    "JPM", "BAC",                                    # Banques
    "LVMUY"                                          # LVMH (ADR US)
]

class MarketScreener:
    def __init__(self):
        self.base_url = "https://www.alphavantage.co/query"

    def _analyser_technique(self, ticker):
        """Analyse une action et retourne son état (Achat/Vente/Neutre)"""
        try:
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": ticker,
                "outputsize": "compact", # Rapide (100 jours)
                "apikey": ALPHA_VANTAGE_KEY
            }
            resp = requests.get(self.base_url, params=params)
            data = resp.json()

            if "Time Series (Daily)" not in data: return None

            # Création DataFrame
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
            df = df.rename(columns={"4. close": "Close"})
            df["Close"] = df["Close"].astype(float)
            df = df.iloc[::-1] # Chrono

            # --- CALCULS INDICATEURS ---
            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=9, adjust=False).mean()

            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # Bollinger
            sma20 = df['Close'].rolling(20).mean()
            std20 = df['Close'].rolling(20).std()
            bollinger_low = sma20 - (std20 * 2)

            # Tendance (SMA 50)
            sma50 = df['Close'].rolling(50).mean()

            # Dernière ligne (Aujourd'hui)
            last = df.iloc[-1]
            idx = df.index[-1] # La date
            
            current_price = last['Close']
            current_rsi = rsi.iloc[-1]
            current_macd = macd.iloc[-1]
            current_sig = signal_line.iloc[-1]
            current_blow = bollinger_low.iloc[-1]
            current_sma50 = sma50.iloc[-1]

            # --- LOGIQUE DE SCORING (DETECTION SIGNAL) ---
            score = 0
            reasons = []

            # 1. Tendance de fond (SMA 50)
            if current_price > current_sma50:
                score += 1
                # reasons.append("Tendance Haussière")
            
            # 2. Momentum (MACD)
            if current_macd > current_sig:
                score += 1
                reasons.append("MACD Vert")
            
            # 3. Point d'entrée (RSI)
            if current_rsi < 40:
                score += 2
                reasons.append(f"RSI Bas ({current_rsi:.1f})")
            elif current_rsi > 70:
                score -= 2
                reasons.append("Surchauffe")

            # 4. Rebond technique (Bollinger)
            if current_price < current_blow * 1.02: # A 2% de la bande basse
                score += 2
                reasons.append("Support Bollinger")

            # VERDICT
            status = "NEUTRE"
            if score >= 3: status = "ACHAT FORT 🚀"
            elif score == 2: status = "ACHAT POTENTIEL ✅"
            elif score <= -1: status = "VENTE ❌"

            return {
                "ticker": ticker,
                "price": current_price,
                "status": status,
                "score": score,
                "details": ", ".join(reasons)
            }

        except Exception as e:
            return None

    def lancer_scan(self):
        print(f"📡 Démarrage du Scanner sur {len(WATCHLIST)} actions...")
        print("⏳ Cela va prendre quelques minutes (Pause de 12s entre chaque scan pour respecter l'API)...")
        print("-" * 60)
        print(f"{'TICKER':<8} | {'PRIX':<10} | {'VERDICT':<20} | {'DETAILS'}")
        print("-" * 60)

        opportunites = []

        for ticker in WATCHLIST:
            result = self._analyser_technique(ticker)
            
            if result:
                # Affichage en temps réel
                color = ""
                if "ACHAT" in result['status']:
                    opportunites.append(result)
                    # On affiche tout de suite
                    print(f"{result['ticker']:<8} | {result['price']:<10.2f} | {result['status']:<20} | {result['details']}")
                else:
                    # Optionnel : Afficher aussi les neutres/ventes ou juste "." pour dire que ça avance
                    print(f"{ticker:<8} ... Pas de signal d'achat.")
            else:
                print(f"{ticker:<8} ... Erreur données.")

            # PAUSE OBLIGATOIRE (API GRATUITE)
            # 5 appels / minute = 1 appel toutes les 12 secondes
            time.sleep(12) 

        print("-" * 60)
        print(f"\n🎯 SCAN TERMINÉ. {len(opportunites)} Opportunités détectées :")
        for op in opportunites:
            print(f"👉 {op['ticker']} ({op['price']}$): {op['status']}")

if __name__ == "__main__":
    screener = MarketScreener()
    screener.lancer_scan()
