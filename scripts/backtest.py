#!/usr/bin/env python3
# scripts/backtest.py
"""Backtester les modèles"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from trading.brain_trader import BrainTrader
from core.utils import setup_logging

logger = setup_logging(__name__)

def backtest(ticker: str, days: int = 365):
    """
    Backtester un ticker
    
    Args:
        ticker: Ticker à tester
        days: Nombre de jours historiques
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"🧪 BACKTEST: {ticker} ({days} jours)")
    logger.info(f"{'='*70}")
    
    # Télécharger données
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if len(df) == 0:
        logger.error(f"❌ Pas de données pour {ticker}")
        return None
    
    # Buy & Hold
    initial_price = float(df['Close'].iloc[0])
    final_price = float(df['Close'].iloc[-1])
    bh_return = ((final_price - initial_price) / initial_price) * 100
    
    logger.info(f"\n📈 Buy & Hold:")
    logger.info(f"   Prix initial: ${initial_price:.2f}")
    logger.info(f"   Prix final: ${final_price:.2f}")
    logger.info(f"   Return: {bh_return:+.2f}%")
    
    # Stratégie IA (simulation simple)
    trader = BrainTrader()
    prediction = trader.predict(ticker)
    
    if prediction:
        logger.info(f"\n🤖 Signal actuel du modèle:")
        logger.info(f"   Action: {prediction['action']}")
        logger.info(f"   Secteur: {prediction['sector']}")
        logger.info(f"   Capital alloué: ${prediction['capital']:,.2f}")
    
    return {
        'ticker': ticker,
        'buy_hold_return': bh_return,
        'current_signal': prediction['action'] if prediction else 'N/A'
    }

def main():
    parser = argparse.ArgumentParser(description='Backtester les modèles')
    
    parser.add_argument(
        'tickers',
        nargs='+',
        help='Tickers à backtester'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Nombre de jours historiques'
    )
    
    args = parser.parse_args()
    
    results = []
    
    for ticker in args.tickers:
        result = backtest(ticker, args.days)
        if result:
            results.append(result)
    
    # Résumé
    if results:
        logger.info("\n" + "="*70)
        logger.info("📊 RÉSUMÉ")
        logger.info("="*70)
        
        df_results = pd.DataFrame(results)
        logger.info(f"\n{df_results.to_string(index=False)}")
        
        avg_return = df_results['buy_hold_return'].mean()
        logger.info(f"\n💰 Return moyen (Buy & Hold): {avg_return:+.2f}%")

if __name__ == "__main__":
    main()
