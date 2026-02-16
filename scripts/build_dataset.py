#!/usr/bin/env python3
# scripts/build_dataset.py
"""
Script de construction de Dataset pour Ploutos V8.
Génère des fichiers CSV standardisés (OHLCV) pour l'entraînement.

Sources gérées :
1. Alpaca (Priorité 1) : Permet 5-10 ans d'historique 1h (si clés API présentes).
2. Yahoo Finance (Fallback) : Limité à 730 jours (2 ans) d'historique 1h.

Usage :
    python scripts/build_dataset.py --period 10y --interval 1h
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from core.utils import setup_logging

logger = setup_logging(__name__, "build_dataset.log")
load_dotenv()

# Liste des tickers (Tech + Indices + Secteurs)
TICKERS = [
    "NVDA",
    "MSFT",
    "AAPL",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",  # Tech Giants
    "SPY",
    "QQQ",
    "VOO",
    "VTI",  # Indices
    "XLE",
    "XLF",
    "XLK",
    "XLV",  # Secteurs
]

DATA_DIR = Path("data/dataset_v8")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_alpaca_data(ticker: str, start_date: str, end_date: str, interval: str):
    """Télécharge via Alpaca (Historique étendu)."""
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_PAPER_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_PAPER_SECRET_KEY")

    if not api_key or not api_secret:
        return None

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError:
        logger.warning(f"Alpaca-py non installé. `pip install alpaca-py` pour l'historique étendu.")
        return None

    client = StockHistoricalDataClient(api_key, api_secret)

    tf = TimeFrame.Hour if interval == "1h" else TimeFrame.Day

    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=tf,
        start=pd.Timestamp(start_date),
        end=pd.Timestamp(end_date),
        adjustment="all",
    )

    logger.info(f"Downloading {ticker} via Alpaca ({start_date} -> {end_date})...")
    try:
        bars = client.get_stock_bars(req)
        if bars.df.empty:
            return None

        df = bars.df.reset_index()
        # Format standard : Date, Open, High, Low, Close, Volume
        df = df.rename(
            columns={
                "timestamp": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df = df.set_index("Date").sort_index()
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        logger.error(f"Alpaca error {ticker}: {e}")
        return None


def get_yahoo_data(ticker: str, period: str, interval: str):
    """Télécharge via Yahoo Finance (Max 2 ans 1h)."""
    logger.info(f"Downloading {ticker} via Yahoo ({period} history)...")
    try:
        # Yahoo gère ses limitations 'period' lui-même
        df = yf.download(
            tickers=ticker, period=period, interval=interval, progress=False, auto_adjust=True
        )
        if df.empty:
            return None

        # Standardisation
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    except Exception as e:
        logger.error(f"Yahoo error {ticker}: {e}")
        return None


def process_ticker(ticker, period, interval):
    """Traite un ticker : Alpaca -> Yahoo -> Sauvegarde."""

    # Calcul des dates (pour Alpaca)
    end_date = datetime.now()
    years = 2
    if period.endswith("y"):
        years = int(period[:-1])

    start_date = end_date - timedelta(days=365 * years)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # 1. Essayer Alpaca (Meilleur historique)
    df = get_alpaca_data(ticker, start_str, end_str, interval)
    source = "Alpaca"

    if df is None:
        # 2. Fallback Yahoo (Max 2 ans 1h)
        source = "Yahoo"
        # Ajuster period si 1h
        adjusted_period = period
        if interval == "1h" and years > 2:
            logger.warning(f"Yahoo limite 1h à 2 ans. Ajustement {period} -> 2y.")
            adjusted_period = "2y"

        df = get_yahoo_data(ticker, adjusted_period, interval)

    if df is None or df.empty:
        logger.error(f"❌ Failed to download {ticker}")
        return

    # Nettoyage
    df = df.ffill().bfill()  # Remplir trous

    # Validation basique
    if interval == "1h" and len(df) < 1000:
        logger.warning(f"⚠️  {ticker}: Peu de données ({len(df)} lignes) via {source}")

    # Sauvegarde
    filename = DATA_DIR / f"{ticker}_{interval}.csv"
    df.to_csv(filename)
    logger.info(f"✅ Saved {ticker} into {filename} via {source} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Ploutos Dataset Builder")
    parser.add_argument("--period", type=str, default="2y", help="Ex: 2y, 5y, 10y")
    parser.add_argument("--interval", type=str, default="1h", help="Ex: 1h, 1d")
    args = parser.parse_args()

    # Créer dossier
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building dataset: Period={args.period}, Interval={args.interval}")
    logger.info(f"Target Directory: {DATA_DIR.resolve()}")

    # Vérification API
    ak = os.getenv("ALPACA_API_KEY") or os.getenv("ALPACA_PAPER_API_KEY")
    ask = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_PAPER_SECRET_KEY")

    if ak and ask:
        logger.info(f"KEYS DETECTED: Alpaca keys found. Mode 'Full History' enabled.")
    else:
        logger.warning(f"NO KEYS: Alpaca keys not found. Mode 'Yahoo Limited' (Max 2y for 1h)")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(process_ticker, ticker, args.period, args.interval)
            for ticker in TICKERS
        ]
        # Attendre fin
        for f in futures:
            f.result()

    logger.info("Dataset generation complete.")


if __name__ == "__main__":
    main()
