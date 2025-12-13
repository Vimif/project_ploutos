#!/usr/bin/env python3
"""
Fetch Full History for All Ploutos Tickers
Downloads daily data for all configured sectors and merges into a single CSV.
"""

import os
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
TICKERS = {
    "GROWTH": ["NVDA", "MSFT", "AAPL", "GOOGL", "AMZN"],
    "DEFENSIVE": ["SPY", "QQQ", "VOO", "VTI"],
    "ENERGY": ["XOM", "CVX", "COP", "XLE"],
    "FINANCE": ["JPM", "BAC", "WFC", "GS"]
}

OUTPUT_FILE = "data/multi_ticker_history.csv"

def fetch_data():
    all_data = []
    
    # Calculate start date (5 years ago to have enough data)
    start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"📅 Fetching data from {start_date} to {end_date}")
    
    for sector, symbols in TICKERS.items():
        logger.info(f"🚀 Processing sector: {sector}")
        
        for symbol in symbols:
            try:
                logger.info(f"   ⬇️ Downloading {symbol}...")
                
                # Download with minimal parameters
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                
                if df is None or df.empty:
                    logger.warning(f"   ⚠️ No data for {symbol}")
                    continue
                
                # Reset index to make Date a column
                df = df.reset_index()
                
                # Rename columns to standard names
                df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
                
                # Keep only OHLCV
                df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
                
                # Add metadata
                df['Ticker'] = symbol
                df['Sector'] = sector
                
                # Remove any NaN rows
                df = df.dropna()
                
                if len(df) == 0:
                    logger.warning(f"   ⚠️ No valid data for {symbol}")
                    continue
                
                all_data.append(df)
                logger.info(f"   ✅ {symbol}: {len(df)} rows")
                
            except Exception as e:
                logger.error(f"   ❌ Error fetching {symbol}: {type(e).__name__}: {e}")

    if not all_data:
        logger.error("❌ No data fetched!")
        return

    # Merge all
    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    # Save
    os.makedirs('data', exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    logger.info("="*50)
    logger.info(f"🎉 SUCCESS! Data saved to {OUTPUT_FILE}")
    logger.info(f"📊 Total Rows: {len(final_df)}")
    logger.info(f"📈 Tickers: {final_df['Ticker'].nunique()}")
    logger.info(f"📊 Date Range: {final_df['Date'].min()} to {final_df['Date'].max()}")
    logger.info(f"📊 Sectors: {', '.join(final_df['Sector'].unique())}")
    logger.info("="*50)

if __name__ == '__main__':
    fetch_data()
