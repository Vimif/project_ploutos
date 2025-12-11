#!/usr/bin/env python3
"""
Download Historical Data Script (Fixed)
=======================================

Télécharge les données historiques via yfinance avec une période ajustée
pour respecter la limite stricte de 730 jours de Yahoo Finance pour les données horaires.

Usage:
    python scripts/download_data.py
"""

import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Configuration du projet
TICKERS = [
    # GROWTH
    "NVDA", "MSFT", "AAPL", "GOOGL", "AMZN",
    # DEFENSIVE
    "SPY", "QQQ", "VOO", "VTI",
    # ENERGY
    "XOM", "CVX", "COP", "XLE",
    # FINANCE
    "JPM", "BAC", "WFC", "GS"
]

# FIX: Yahoo Finance est très strict sur la limite de 730 jours pour les données 1h.
# On prend 720 jours pour être sûr (marge de sécurité).
DAYS_BACK = 720
START_DATE = (datetime.now() - timedelta(days=DAYS_BACK)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')
OUTPUT_FILE = "data/historical_daily.csv"
INTERVAL = "1h"

def download_and_process():
    print(f"🚀 Démarrage du téléchargement pour {len(TICKERS)} tickers...")
    print(f"📅 Période: {START_DATE} à {END_DATE} ({DAYS_BACK} jours)")
    print(f"⏱️  Intervalle: {INTERVAL}")

    os.makedirs("data", exist_ok=True)

    all_data = []

    for ticker in TICKERS:
        print(f"  ⬇️  Downloading {ticker}...", end=" ", flush=True)
        try:
            # FIX: auto_adjust=True pour éviter le warning
            df = yf.download(
                ticker, 
                start=START_DATE, 
                end=END_DATE, 
                interval=INTERVAL, 
                progress=False,
                auto_adjust=True
            )
            
            if len(df) == 0:
                print(f"❌ Empty data!")
                continue

            # Aplatir le MultiIndex si nécessaire
            if isinstance(df.columns, pd.MultiIndex):
                # Si colonnes sont ('Close', 'NVDA'), on garde juste 'Close'
                # Yahoo a changé son format récemment
                try:
                    df = df.xs(ticker, axis=1, level=1, drop_level=True)
                except:
                    # Fallback si structure différente
                    df.columns = df.columns.get_level_values(0)
            
            # Reset index pour avoir la date en colonne
            df = df.reset_index()
            
            # Renommer la colonne date
            # Yahoo retourne souvent 'Datetime' pour les données intraday
            date_col = None
            for col in df.columns:
                if 'Date' in str(col) or 'Time' in str(col):
                    date_col = col
                    break
            
            if date_col:
                df = df.rename(columns={date_col: 'Date'})
            else:
                print("❌ Date column not found")
                continue
            
            # Ajouter la colonne Ticker
            df['Ticker'] = ticker
            
            # Sélectionner les colonnes nécessaires
            required_cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
            available_cols = [c for c in required_cols if c in df.columns]
            
            if len(available_cols) < 5: # Au moins OHLCV
                print(f"❌ Missing columns. Got: {df.columns}")
                continue
                
            df = df[available_cols]
            
            all_data.append(df)
            print(f"✅ {len(df)} rows")
            
        except Exception as e:
            print(f"❌ Error: {e}")

    if not all_data:
        print("\n❌ ÉCHEC TOTAL: Aucune donnée récupérée.")
        print("💡 Solution de secours: Utilisation de données journalières ('1d') au lieu de horaires ('1h')")
        print("   Cela permet d'avoir plus d'historique et évite la limite de 730 jours.")
        
        retry = input("Voulez-vous essayer avec des données journalières ? (y/n) ")
        if retry.lower() == 'y':
            download_daily_fallback()
        return

    # Concaténation
    print("\n🔄 Fusion des données...")
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Nettoyage
    final_df = final_df.sort_values(['Ticker', 'Date'])
    final_df = final_df.dropna()

    # Sauvegarde
    print(f"💾 Sauvegarde vers {OUTPUT_FILE}...")
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n✅ TÉLÉCHARGEMENT TERMINÉ!")
    print(f"📊 Total lignes: {len(final_df)}")
    print(f"📈 Tickers: {final_df['Ticker'].nunique()}")
    print("\n👉 Vous pouvez maintenant lancer l'entraînement :")
    print(f"   python scripts/train_v6_final.py --data {OUTPUT_FILE}")

def download_daily_fallback():
    """Fallback to daily data if hourly fails"""
    print("\n🚀 Téléchargement données JOURNALIÈRES (Fallback)...")
    # 5 ans d'historique pour daily
    start = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    end = datetime.now().strftime('%Y-%m-%d')
    
    all_data = []
    for ticker in TICKERS:
        print(f"  ⬇️  Downloading {ticker} (1d)...", end=" ")
        try:
            df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df = df.xs(ticker, axis=1, level=1, drop_level=True)
                except:
                    df.columns = df.columns.get_level_values(0)
            
            df = df.reset_index()
            # Rename Date
            col_map = {c: 'Date' for c in df.columns if 'Date' in str(c)}
            df = df.rename(columns=col_map)
            df['Ticker'] = ticker
            all_data.append(df)
            print(f"✅ {len(df)} rows")
        except Exception as e:
            print(f"❌ {e}")
            
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ Sauvegardé dans {OUTPUT_FILE} (Données journalières)")
        print("⚠️  Note: L'entraînement sera moins précis pour l'intraday mais fonctionnera.")

if __name__ == "__main__":
    download_and_process()
