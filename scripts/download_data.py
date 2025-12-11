#!/usr/bin/env python3
"""
Download Historical Data Script
===============================

Télécharge les données historiques via yfinance (plus rapide et complet que Alpaca pour l'historique)
et les formate pour l'entraînement Ploutos V6.

Usage:
    python scripts/download_data.py
"""

import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Configuration du projet (Tiré de ton contexte)
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

START_DATE = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 ans max (limite Yahoo horaire)
END_DATE = datetime.now().strftime('%Y-%m-%d')
OUTPUT_FILE = "data/historical_daily.csv"
INTERVAL = "1h"  # Données horaires comme spécifié dans ton projet

def download_and_process():
    print(f"🚀 Démarrage du téléchargement pour {len(TICKERS)} tickers...")
    print(f"📅 Période: {START_DATE} à {END_DATE}")
    print(f"⏱️  Intervalle: {INTERVAL}")

    # Création du dossier data si inexistant
    os.makedirs("data", exist_ok=True)

    all_data = []

    for ticker in TICKERS:
        print(f"  ⬇️  Downloading {ticker}...", end=" ", flush=True)
        try:
            # Téléchargement
            df = yf.download(ticker, start=START_DATE, end=END_DATE, interval=INTERVAL, progress=False)
            
            if len(df) == 0:
                print(f"❌ Empty data!")
                continue

            # Aplatir le MultiIndex si nécessaire (yfinance retourne parfois des MultiIndex)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Reset index pour avoir la date en colonne
            df = df.reset_index()
            
            # Renommer la colonne date (Datetime ou Date)
            date_col = [c for c in df.columns if 'Date' in c or 'Time' in c][0]
            df = df.rename(columns={date_col: 'Date'})
            
            # Ajouter la colonne Ticker
            df['Ticker'] = ticker
            
            # Sélectionner et ordonner les colonnes
            cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
            df = df[cols]
            
            all_data.append(df)
            print(f"✅ {len(df)} rows")
            
        except Exception as e:
            print(f"❌ Error: {e}")

    if not all_data:
        print("❌ Aucune donnée téléchargée. Vérifiez votre connexion internet.")
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

if __name__ == "__main__":
    download_and_process()
