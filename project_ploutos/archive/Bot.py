import yfinance as yf
import pandas as pd
import numpy as np
import itertools

# --- CONFIGURATION ---
ACTION = "IBM"
PERIODE_ENTRAINEMENT = "5y" # On s'entraîne sur 5 ans d'historique !

class BigDataTrainer:
    def __init__(self):
        pass

    def recuperer_donnees_massives(self, ticker):
        print(f"📥 Téléchargement de {PERIODE_ENTRAINEMENT} d'historique pour {ticker} via Yahoo Finance...")
        
        # C'est ici que la magie opère : yfinance télécharge tout gratuitement
        df = yf.download(ticker, period=PERIODE_ENTRAINEMENT, interval="1d", progress=False)
        
        if df.empty:
            return None
            
        # Nettoyage des données (Yahoo renvoie parfois des MultiIndex)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.rename(columns={"Close": "Close", "Open": "Open", "High": "High", "Low": "Low"})
        
        # Calcul des indicateurs sur 5 ANS
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df.dropna()

    def simulation_rapide(self, df, params):
        """Simule le trading avec une combinaison de paramètres"""
        rsi_buy, rsi_sell, stop_loss_pct = params
        
        capital = 10000
        position = 0
        prix_achat = 0
        nb_trades = 0
        
        # Conversion en numpy array pour aller 100x plus vite (Astuce Pro)
        closes = df['Close'].values
        rsis = df['RSI'].values
        macds = df['MACD'].values
        sigs = df['Signal_Line'].values
        
        for i in range(len(closes)):
            prix = closes[i]
            rsi = rsis[i]
            
            # VENTE
            if position > 0:
                # Stop Loss
                if (prix - prix_achat) / prix_achat < -stop_loss_pct:
                    capital += position * prix
                    position = 0
                    nb_trades += 1
                    continue
                
                # Signal Technique Vente
                if rsi > rsi_sell:
                    capital += position * prix
                    position = 0
                    nb_trades += 1
                    continue

            # ACHAT
            if position == 0:
                # Achat si MACD Croissant ET RSI bas
                if macds[i] > sigs[i] and rsi < rsi_buy:
                    position = capital / prix # On achète des fractions (plus simple pour le test)
                    capital = 0
                    prix_achat = prix
                    nb_trades += 1

        valeur_finale = capital + (position * closes[-1])
        return valeur_finale, nb_trades

    def lancer_optimisation(self, ticker):
        df = self.recuperer_donnees_massives(ticker)
        if df is None:
            print("❌ Erreur de téléchargement Yahoo.")
            return

        print(f"🧠 Entraînement de l'IA sur {len(df)} jours de bourse...")
        print("⚙️ Recherche de la meilleure stratégie...")

        # Grille de recherche (Grid Search)
        # On teste beaucoup plus de combinaisons car on a plus de données
        rsi_buys = [20, 25, 30, 35, 40, 45]
        rsi_sells = [60, 65, 70, 75, 80]
        stop_losses = [0.03, 0.05, 0.07, 0.10, 0.15]
        
        combinaisons = list(itertools.product(rsi_buys, rsi_sells, stop_losses))
        
        best_score = 0
        best_params = None
        best_trades = 0
        
        for params in combinaisons:
            resultat, trades = self.simulation_rapide(df, params)
            
            # Critère de succès : On veut gagner de l'argent MAIS en faisant au moins 5 trades
            # (pour éviter les coups de chance uniques)
            if resultat > best_score and trades > 5:
                best_score = resultat
                best_params = params
                best_trades = trades
        
        # Comparaison avec le marché (Buy & Hold)
        prix_debut = df['Close'].iloc[0]
        prix_fin = df['Close'].iloc[-1]
        perf_marche = (prix_fin - prix_debut) / prix_debut * 100
        perf_bot = (best_score - 10000) / 10000 * 100

        print("\n🏆 --- RÉSULTATS DE L'OPTIMISATION (5 ANS) ---")
        print(f"Action : {ticker}")
        print(f"Performance Marché (Buy&Hold) : {perf_marche:+.2f}%")
        print(f"Performance IA Optimisée      : {perf_bot:+.2f}%")
        print("-" * 40)
        
        if best_params:
            print("✅ STRATÉGIE GAGNANTE TROUVÉE :")
            print(f"  1. Acheter quand RSI < {best_params[0]}")
            print(f"  2. Vendre quand RSI > {best_params[1]}")
            print(f"  3. Stop Loss à -{best_params[2]*100}%")
            print(f"  (Basé sur {best_trades} trades simulés)")
        else:
            print("❌ Aucune stratégie stable trouvée (Marché trop difficile).")

if __name__ == "__main__":
    bot = BigDataTrainer()
    bot.lancer_optimisation(ACTION)
