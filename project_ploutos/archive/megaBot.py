import yfinance as yf
import requests
import pandas as pd
import numpy as np
import itertools
import time

# --- CONFIGURATION ---
ALPHA_VANTAGE_KEY = "VTPL0D2304CV9GHF"
WATCHLIST = ["IBM", "AAPL", "TSLA", "MSFT", "NVDA", "AMD", "INTC", "AMZN", "GOOGL"]

class MegaBot:
    def __init__(self):
        self.base_url = "https://www.alphavantage.co/query"

    def _optimiser_action(self, ticker):
        """Trouve les meilleurs seuils RSI (Yahoo Finance)"""
        try:
            # Force une nouvelle session propre pour éviter les conflits
            df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            
            # Gestion des formats Yahoo bizarres
            if df.empty: return (30, 70)
            if isinstance(df.columns, pd.MultiIndex):
                try: df.columns = df.columns.get_level_values(0)
                except: pass
            if 'Close' not in df.columns:
                if len(df.columns) > 0: df['Close'] = df.iloc[:, 0]
                else: return (30, 70)

            # Calcul RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            df = df.dropna()

            if len(df) < 30: return (30, 70)

            # Optimisation Ultra-Rapide
            closes = df['Close'].values
            rsis = df['RSI'].values
            best_perf = -99999
            best_params = (30, 70)

            for rb in [25, 30, 35, 40, 45]:
                for rs in [65, 70, 75, 80]:
                    # Simulation vectorisée simplifiée
                    capital = 1000
                    pos = 0
                    for i in range(len(closes)):
                        if pos == 0 and rsis[i] < rb:
                            pos = capital / closes[i]
                            capital = 0
                        elif pos > 0 and rsis[i] > rs:
                            capital = pos * closes[i]
                            pos = 0
                    val = capital + (pos * closes[-1])
                    if val > best_perf:
                        best_perf = val
                        best_params = (rb, rs)
            
            return best_params
        except:
            return (30, 70)

    def _analyser_action(self, ticker, params):
        """Analyse temps réel avec Alpha Vantage (avec Retry)"""
        rsi_buy, rsi_sell = params
        
        # On tente jusqu'à 2 fois en cas d'erreur réseau
        for tentative in range(2):
            try:
                req_params = {
                    "function": "TIME_SERIES_DAILY",
                    "symbol": ticker,
                    "outputsize": "compact",
                    "apikey": ALPHA_VANTAGE_KEY
                }
                
                # Timeout de 10s pour ne pas bloquer indéfiniment
                resp = requests.get(self.base_url, params=req_params, timeout=10)
                data = resp.json()
                
                if "Note" in data: 
                    print(f"   ⚠️ Limite API. Pause de 60s...")
                    time.sleep(60)
                    continue # On réessaie après la pause

                if "Time Series (Daily)" not in data: 
                    # Si c'est la dernière tentative, on échoue
                    if tentative == 1: return None
                    time.sleep(2)
                    continue

                df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
                df = df.rename(columns={"4. close": "Close"})
                df["Close"] = df["Close"].astype(float)
                df = df.iloc[::-1]
                
                # Indicateurs
                df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
                df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
                last = df.iloc[-1]
                
                status = "NEUTRE"
                score = 0
                
                if last['RSI'] < rsi_buy:
                    status = "ACHAT (RSI)"
                    score = 1
                    if last['MACD'] > last['Signal_Line']:
                        status = "ACHAT FORT 🚀"
                        score = 2
                elif last['RSI'] > rsi_sell:
                    status = "VENTE 🔴"
                    score = -1
                    
                return {
                    "ticker": ticker,
                    "price": last['Close'],
                    "rsi": last['RSI'],
                    "params": params,
                    "status": status,
                    "score": score
                }
            except Exception as e:
                time.sleep(2)
                continue
        
        return None

    def lancer_mega_scan(self):
        print(f"🛡️ Démarrage de la Station de Trading ({len(WATCHLIST)} actions)...")
        print("⏳ Le système va optimiser sa stratégie pour CHAQUE action. Patientez...")
        print("="*90)
        print(f"{'ACTION':<6} | {'PRIX':<8} | {'RSI ACTUEL':<10} | {'SEUILS IA':<12} | {'VERDICT'}")
        print("="*90)
        
        opportunites = []
        
        for ticker in WATCHLIST:
            # 1. Optimisation (Yahoo)
            best_params = self._optimiser_action(ticker)
            
            # Petite pause technique pour laisser le réseau respirer
            time.sleep(1)
            
            # 2. Analyse (Alpha Vantage)
            result = self._analyser_action(ticker, best_params)
            
            if result:
                seuil_str = f"<{result['params'][0]} / >{result['params'][1]}"
                print(f"{ticker:<6} | {result['price']:<8.2f} | {result['rsi']:<10.1f} | {seuil_str:<12} | {result['status']}")
                if result['score'] >= 1:
                    opportunites.append(result)
            else:
                print(f"{ticker:<6} ... Données indisponibles.")

            # Pause OBLIGATOIRE (API Gratuite = 5 appels / min max)
            # On met 15 secondes pour être large
            time.sleep(15) 

        print("="*90)
        print(f"\n🎯 RÉSUMÉ FINAL : {len(opportunites)} Opportunités détectées")
        for op in opportunites:
            print(f"👉 {op['ticker']} ({op['price']}$): {op['status']}")

if __name__ == "__main__":
    bot = MegaBot()
    bot.lancer_mega_scan()
