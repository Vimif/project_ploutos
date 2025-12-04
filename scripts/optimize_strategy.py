import sys
import os
# Ajouter le dossier parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- PARAMÈTRES À TESTER ---
STOP_LOSS_OPTIONS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
TAKE_PROFIT_OPTIONS = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
# ---------------------------

def get_historical_data(symbol, days=30):
    """Récupère les données horaires"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        # auto_adjust=True pour éviter le Warning
        df = yf.download(symbol, start=start_date, end=end_date, interval="1h", progress=False, auto_adjust=True)
        
        # Nettoyage si MultiIndex (cas courant avec les nouvelles versions)
        if isinstance(df.columns, pd.MultiIndex):
            # On garde seulement le niveau 'Price' ou le ticker, on veut juste 'Close' propre
            try:
                df = df.xs(symbol, axis=1, level=1)
            except:
                pass # Parfois la structure est différente, on gère plus bas
                
        return df
    except Exception as e:
        print(f"❌ Erreur data {symbol}: {e}")
        return None

def backtest_single_pair(df, sl_pct, tp_pct):
    """Simule une stratégie simple"""
    if df is None or len(df) == 0:
        return 0
        
    # Récupérer la série de prix proprement
    try:
        # Si c'est un DataFrame avec une colonne 'Close', on la prend
        if 'Close' in df.columns:
            prices = df['Close']
        # Sinon c'est peut-être déjà une Series
        else:
            prices = df.iloc[:, 0] # On prend la première colonne par défaut
            
        # S'assurer que c'est bien une série numérique 1D
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
            
        prices = prices.to_numpy() # Conversion en numpy pour performance et éviter les erreurs d'index
        
    except Exception as e:
        # print(f"Debug format error: {e}")
        return 0

    if len(prices) < 2:
        return 0
        
    entry_price = float(prices[0])
    
    for current_price in prices:
        current_price = float(current_price)
        pnl_pct = (current_price - entry_price) / entry_price
        
        # Touché Stop Loss ?
        if pnl_pct <= -sl_pct:
            return -sl_pct
            
        # Touché Take Profit ?
        if pnl_pct >= tp_pct:
            return tp_pct
            
    # Si ni l'un ni l'autre à la fin
    final_pnl = (float(prices[-1]) - entry_price) / entry_price
    return final_pnl

def optimize():
    print("🧠 DÉMARRAGE DE L'OPTIMISATION (Simulation sur les Top Tickers)...")
    
    TEST_TICKERS = ['NVDA', 'AAPL', 'MSFT', 'TSLA', 'AMD', 'AMZN', 'GOOGL']
    
    print("📥 Téléchargement des données historiques (30 jours, 1h)...")
    market_data = {}
    for ticker in TEST_TICKERS:
        df = get_historical_data(ticker)
        if df is not None and not df.empty:
            market_data[ticker] = df
            
    best_score = -999
    best_params = (0, 0)
    results = []

    print(f"\n🧪 Test de {len(STOP_LOSS_OPTIONS) * len(TAKE_PROFIT_OPTIONS)} combinaisons...")
    
    count = 0
    for sl in STOP_LOSS_OPTIONS:
        for tp in TAKE_PROFIT_OPTIONS:
            total_pnl = 0
            valid_tickers = 0
            
            for ticker, df in market_data.items():
                pnl = backtest_single_pair(df, sl, tp)
                total_pnl += pnl
                valid_tickers += 1
            
            if valid_tickers > 0:
                avg_score = (total_pnl / valid_tickers) * 100
                results.append((sl, tp, avg_score))
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = (sl, tp)
            
            count += 1
            if count % 5 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()

    print("\n\n🏆 RÉSULTATS DE L'OPTIMISATION (Sur 30 jours)")
    print("="*40)
    print(f"MEILLEURS PARAMÈTRES TROUVÉS :")
    print(f"🛑 Stop Loss   : {best_params[0]*100:.1f}%")
    print(f"🎯 Take Profit : {best_params[1]*100:.1f}%")
    print(f"💰 Rendement Moyen Théorique : {best_score:.2f}% par trade")
    print("="*40)
    
    print("\nTOP 3 DES COMBINAISONS :")
    sorted_results = sorted(results, key=lambda x: x[2], reverse=True)[:3]
    for i, (sl, tp, score) in enumerate(sorted_results):
        print(f"{i+1}. SL: {sl*100:.1f}% / TP: {tp*100:.1f}% -> {score:.2f}%")

if __name__ == "__main__":
    optimize()