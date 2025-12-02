import requests
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
ALPHA_VANTAGE_KEY = "N7Y23TIA23OZ892L" 
CAPITAL_INITIAL = 10000
TICKER_CIBLE = "IBM"

# --- PARAMETRES OPTIMISÉS PAR L'IA (Issus de votre entraînement) ---
# Ces valeurs viennent du résultat de l'optimisation sur 5 ans
RSI_ACHAT_OPTIMAL = 45   # L'IA a trouvé que 45 était mieux que 30 pour IBM
RSI_VENTE_OPTIMAL = 80   # L'IA suggère d'attendre 80 pour vendre
STOP_LOSS_PCT = 0.05     # Stop Loss à 5%

class OptimizedTradingBot:
    def __init__(self):
        self.base_url = "https://www.alphavantage.co/query"

    def _calculer_indicateurs(self, df):
        """Calcule les indicateurs techniques nécessaires"""
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bandes Bollinger
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['STD_20'] = df['Close'].rolling(window=20).std()
        df['Bollinger_Low'] = df['SMA_20'] - (df['STD_20'] * 2)

        # Tendance (SMA 50 pour le mode compact)
        df['TREND_SMA'] = df['Close'].rolling(window=50).mean()
        
        return df.dropna()

    def executer_strategie(self, ticker):
        print(f"\n🚀 Lancement du Bot Optimisé sur {ticker}...")
        print(f"⚙️ Paramètres IA : Achat < {RSI_ACHAT_OPTIMAL} | Vente > {RSI_VENTE_OPTIMAL} | SL -{STOP_LOSS_PCT*100}%")
        
        # 1. Récupération des données (Mode Compact pour rapidité/stabilité)
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": ticker,
            "outputsize": "compact", 
            "apikey": ALPHA_VANTAGE_KEY
        }
        
        try:
            resp = requests.get(self.base_url, params=params)
            data = resp.json()

            if "Time Series (Daily)" not in data:
                print("❌ Erreur API (Clé ou Limite).")
                return

            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient='index')
            df = df.rename(columns={"4. close": "Close"})
            df["Close"] = df["Close"].astype(float)
            df = df.iloc[::-1] 
            
            # Calculs
            df = self._calculer_indicateurs(df)
            last = df.iloc[-1] # Données d'aujourd'hui
            
            print(f"✅ Données analysées. Prix actuel : {last['Close']:.2f} $")

            # --- MOTEUR DE DÉCISION TEMPS RÉEL ---
            # C'est ici qu'on applique les règles trouvées par l'IA
            
            signal = "NEUTRE"
            raison = ""
            
            # 1. Vérification de VENTE (Est-ce qu'on doit sortir ?)
            if last['RSI'] > RSI_VENTE_OPTIMAL:
                signal = "VENTE 🔴"
                raison = f"RSI ({last['RSI']:.1f}) supérieur au seuil ({RSI_VENTE_OPTIMAL})"
            
            # 2. Vérification d'ACHAT
            elif last['RSI'] < RSI_ACHAT_OPTIMAL:
                # Condition supplémentaire : MACD pour confirmer (Sécurité)
                if last['MACD'] > last['Signal_Line']:
                    signal = "ACHAT 🟢"
                    raison = f"RSI ({last['RSI']:.1f}) inférieur au seuil ({RSI_ACHAT_OPTIMAL}) + MACD Validé"
                else:
                    signal = "ATTENTE ⚪"
                    raison = f"RSI bon ({last['RSI']:.1f}) mais MACD encore négatif"
            
            else:
                signal = "NEUTRE ⚪"
                raison = f"RSI ({last['RSI']:.1f}) dans la zone normale ({RSI_ACHAT_OPTIMAL}-{RSI_VENTE_OPTIMAL})"

            print("\n🔮 --- CONSEIL DU JOUR ---")
            print(f"Date   : {last.name}")
            print(f"Action : {ticker}")
            print(f"Signal : {signal}")
            print(f"Raison : {raison}")
            
            # Résumé des indicateurs clés
            print("\n📊 Tableau de Bord Technique :")
            print(f"   • RSI : {last['RSI']:.1f} (Zone neutre: {RSI_ACHAT_OPTIMAL}-{RSI_VENTE_OPTIMAL})")
            print(f"   • MACD : {'Positif' if last['MACD'] > last['Signal_Line'] else 'Négatif'}")
            print(f"   • Tendance (SMA 50) : {'Haussière' if last['Close'] > last['TREND_SMA'] else 'Baissière'}")

        except Exception as e:
            print(f"❌ Erreur critique : {e}")

if __name__ == "__main__":
    bot = OptimizedTradingBot()
    bot.executer_strategie(TICKER_CIBLE)
