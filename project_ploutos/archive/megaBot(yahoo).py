import yfinance as yf
import pandas as pd
import numpy as np
import time

# --- CONFIGURATION ---
# Plus besoin de clé API Alpha Vantage !
WATCHLIST = ["IBM", "AAPL", "TSLA", "MSFT", "NVDA", "AMD", "INTC", "AMZN", "GOOGL", "META", "NFLX"]

class YahooBot:
    def __init__(self):
        pass

    def _calculer_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _optimiser_et_analyser(self, ticker):
        try:
            # On télécharge tout d'un coup (2 ans)
            # auto_adjust=True est important pour Yahoo
            df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
            
            if df.empty: return None

            # Nettoyage des colonnes MultiIndex (le grand classique de Yahoo)
            if isinstance(df.columns, pd.MultiIndex):
                try: df.columns = df.columns.get_level_values(0)
                except: pass
            
            # Vérif finale colonne
            if 'Close' not in df.columns:
                if len(df.columns) > 0: df['Close'] = df.iloc[:, 0]
                else: return None

            # Calculs Indicateurs
            df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
            df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
            df['RSI'] = self._calculer_rsi(df['Close'])
            
            # On supprime les NaN du début
            df = df.dropna()
            if len(df) < 50: return None

            # --- 1. PARTIE OPTIMISATION (Sur le passé) ---
            # On prend tout SAUF les 5 derniers jours pour l'entraînement
            train_df = df.iloc[:-5] 
            
            closes = train_df['Close'].values
            rsis = train_df['RSI'].values
            
            best_perf = -99999
            best_params = (30, 70) # Défaut

            # Grid Search Rapide
            for rb in [25, 30, 35, 40, 45]:
                for rs in [65, 70, 75, 80]:
                    capital = 1000
                    pos = 0
                    # Simulation vectorisée simplifiée
                    mask_buy = rsis < rb
                    mask_sell = rsis > rs
                    
                    for i in range(len(closes)):
                        if pos == 0 and mask_buy[i]:
                            pos = capital / closes[i]
                            capital = 0
                        elif pos > 0 and mask_sell[i]:
                            capital = pos * closes[i]
                            pos = 0
                    
                    val = capital + (pos * closes[-1])
                    if val > best_perf:
                        best_perf = val
                        best_params = (rb, rs)

            # --- 2. PARTIE ANALYSE (Sur aujourd'hui) ---
            last = df.iloc[-1]
            rsi_buy, rsi_sell = best_params
            
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
                "params": best_params,
                "status": status,
                "score": score
            }

        except Exception as e:
            # print(f"Erreur {ticker}: {e}")
            return None

    def lancer_scan_rapide(self):
        print(f"🚀 Démarrage du SCANNER RAPIDE (Source: Yahoo Finance)...")
        print(f"📋 Liste : {len(WATCHLIST)} actions")
        print("="*95)
        print(f"{'ACTION':<6} | {'PRIX':<8} | {'RSI ACTUEL':<10} | {'SEUILS IA':<12} | {'VERDICT'}")
        print("="*95)
        
        opportunites = []
        
        for ticker in WATCHLIST:
            result = self._optimiser_et_analyser(ticker)
            
            if result:
                seuil_str = f"<{result['params'][0]} / >{result['params'][1]}"
                # Couleur simple pour le terminal
                verdict = result['status']
                
                print(f"{ticker:<6} | {result['price']:<8.2f} | {result['rsi']:<10.1f} | {seuil_str:<12} | {verdict}")
                
                if result['score'] >= 1:
                    opportunites.append(result)
            else:
                print(f"{ticker:<6} ... Erreur données.")
            
            # Pas besoin de pause de 15s ici ! Yahoo est tolérant.
            # Juste une petite pause de courtoisie
            time.sleep(0.5)

        print("="*95)
        print(f"\n🎯 RÉSULTAT : {len(opportunites)} Opportunités trouvées")
        for op in opportunites:
            print(f"👉 {op['ticker']} ({op['price']:.2f}$): {op['status']}")

if __name__ == "__main__":
    bot = YahooBot()
    bot.lancer_scan_rapide()
