import sys
import os
# Ajouter le dossier parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- PARAMÈTRES À TESTER ---
STOP_LOSS_OPTIONS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]  # De 2% à 8%
TAKE_PROFIT_OPTIONS = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]      # De 10% à 25%
# ---------------------------

def get_historical_data(symbol, days=30):
    """Récupère les données horaires pour plus de précision"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df = yf.download(symbol, start=start_date, end=end_date, interval="1h", progress=False)
        return df
    except Exception as e:
        print(f"❌ Erreur data {symbol}: {e}")
        return None

def backtest_single_pair(df, sl_pct, tp_pct):
    """Simule une stratégie simple : Achat au début, gestion SL/TP"""
    # Simplification : On suppose qu'on achète au premier point disponible
    # Dans une vraie opti, on utiliserait les points d'entrée de ton Bot (depuis la BDD)
    if df is None or len(df) == 0:
        return 0
        
    entry_price = df.iloc[0]['Close']  # On achète arbitrairement au début de la période
    # Note: Pour faire mieux, il faudrait lire la table 'predictions' pour savoir QUAND le bot a acheté.
    
    for index, row in df.iterrows():
        current_price = row['Close']
        pnl_pct = (current_price - entry_price) / entry_price
        
        # Touché Stop Loss ?
        if pnl_pct <= -sl_pct:
            return -sl_pct
            
        # Touché Take Profit ?
        if pnl_pct >= tp_pct:
            return tp_pct
            
    # Si ni l'un ni l'autre à la fin
    final_pnl = (df.iloc[-1]['Close'] - entry_price) / entry_price
    return final_pnl

def optimize():
    print("🧠 DÉMARRAGE DE L'OPTIMISATION (Simulation sur les Top Tickers)...")
    
    # Liste des tickers que tu surveilles (Extrait de ton config.tickers)
    TEST_TICKERS = ['NVDA', 'AAPL', 'MSFT', 'TSLA', 'AMD', 'AMZN', 'GOOGL']
    
    # Télécharger les données une fois pour toutes
    print("📥 Téléchargement des données historiques (30 jours, 1h)...")
    market_data = {}
    for ticker in TEST_TICKERS:
        df = get_historical_data(ticker)
        if df is not None:
            market_data[ticker] = df
    
    best_score = -999
    best_params = (0, 0)
    results = []

    print(f"\n🧪 Test de {len(STOP_LOSS_OPTIONS) * len(TAKE_PROFIT_OPTIONS)} combinaisons...")
    
    for sl in STOP_LOSS_OPTIONS:
        for tp in TAKE_PROFIT_OPTIONS:
            total_pnl = 0
            
            for ticker, df in market_data.items():
                # On lance un backtest simple
                pnl = backtest_single_pair(df, sl, tp)
                total_pnl += pnl
            
            # Score moyen par trade simulé
            avg_score = (total_pnl / len(TEST_TICKERS)) * 100
            results.append((sl, tp, avg_score))
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = (sl, tp)
                
            # Barre de progression simple
            sys.stdout.write(".")
            sys.stdout.flush()

    print("\n\n🏆 RÉSULTATS DE L'OPTIMISATION")
    print("="*40)
    print(f"MEILLEURS PARAMÈTRES TROUVÉS :")
    print(f"🛑 Stop Loss   : {best_params[0]*100:.1f}%")
    print(f"🎯 Take Profit : {best_params[1]*100:.1f}%")
    print(f"💰 Rendement Moyen Théorique : {best_score:.2f}% par trade")
    print("="*40)
    
    print("\nTOP 3 DES COMBINAISONS :")
    # Trier par score décroissant
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)[:3]
    for i, (sl, tp, score) in enumerate(sorted_results):
        print(f"{i+1}. SL: {sl*100:.1f}% / TP: {tp*100:.1f}% -> {score:.2f}%")

if __name__ == "__main__":
    optimize()