#!/usr/bin/env python3
# scripts/run_trader.py
"""Lancer le trader autonome"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
from datetime import datetime
from trading.brain_trader import BrainTrader
from trading.portfolio import Portfolio
from core.utils import setup_logging
from config.settings import TRADING_CONFIG
import yfinance as yf

logger = setup_logging(__name__, 'trader.log')

def get_current_prices(tickers: list):
    """Obtenir les prix actuels"""
    prices = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, period='1d', progress=False)
            if len(data) > 0:
                prices[ticker] = float(data['Close'].iloc[-1])
        except:
            pass
    return prices

def run_trading_cycle(trader: BrainTrader, portfolio: Portfolio):
    """Exécuter un cycle de trading"""
    logger.info("\n" + "="*70)
    logger.info(f"🔄 CYCLE DE TRADING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)
    
    # Obtenir prédictions
    predictions = trader.predict_all()
    
    # Collecter tous les tickers
    all_tickers = []
    for sector_preds in predictions.values():
        for pred in sector_preds:
            all_tickers.append(pred['ticker'])
    
    # Obtenir prix actuels
    current_prices = get_current_prices(all_tickers)
    
    # Exécuter les trades
    actions_taken = {'buy': 0, 'sell': 0, 'hold': 0}
    
    for sector, sector_preds in predictions.items():
        logger.info(f"\n🧠 {sector.upper()}:")
        
        for pred in sector_preds:
            ticker = pred['ticker']
            action = pred['action']
            price = current_prices.get(ticker)
            
            if price is None:
                logger.warning(f"  ⚠️  {ticker}: Prix indisponible")
                continue
            
            emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '⚪'}[action]
            logger.info(f"  {emoji} {ticker}: {action} @ ${price:.2f}")
            
            if action == 'BUY':
                # Acheter
                amount = pred['capital']
                portfolio.buy(ticker, price, amount)
                actions_taken['buy'] += 1
                
            elif action == 'SELL':
                # Vendre si position existante
                if portfolio.get_position(ticker):
                    portfolio.sell(ticker, price)
                    actions_taken['sell'] += 1
            else:
                actions_taken['hold'] += 1
    
    # Résumé
    summary = portfolio.get_summary(current_prices)
    
    logger.info("\n" + "="*70)
    logger.info("📊 RÉSUMÉ DU CYCLE")
    logger.info("="*70)
    logger.info(f"🎯 Actions: {actions_taken['buy']} BUY | {actions_taken['sell']} SELL | {actions_taken['hold']} HOLD")
    logger.info(f"💰 Cash: ${summary['cash']:,.2f}")
    logger.info(f"📈 Valeur totale: ${summary['total_value']:,.2f}")
    logger.info(f"💵 P&L: ${summary['total_return']:+,.2f} ({summary['total_return_pct']:+.2f}%)")
    logger.info(f"🔢 Positions: {summary['positions_count']}")
    
    # Sauvegarder
    portfolio.save_state()
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='Lancer le trader autonome')
    
    parser.add_argument(
        '--capital',
        type=float,
        default=TRADING_CONFIG['initial_capital'],
        help='Capital initial'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=TRADING_CONFIG['check_interval_minutes'],
        help='Intervalle entre les cycles (minutes)'
    )
    
    parser.add_argument(
        '--cycles',
        type=int,
        default=None,
        help='Nombre de cycles (illimité par défaut)'
    )
    
    parser.add_argument(
        '--paper',
        action='store_true',
        help='Mode paper trading (simulation)'
    )
    
    args = parser.parse_args()
    
    logger.info("\n" + "="*70)
    logger.info("🤖 DÉMARRAGE DU TRADER AUTONOME")
    logger.info("="*70)
    logger.info(f"💰 Capital: ${args.capital:,.2f}")
    logger.info(f"⏱️  Intervalle: {args.interval} minutes")
    logger.info(f"📊 Mode: {'Paper Trading' if args.paper else 'LIVE'}")
    
    if not args.paper:
        logger.warning("⚠️  MODE LIVE - TRADES RÉELS !")
        response = input("Continuer? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("❌ Annulé")
            return
    
    # Initialiser
    trader = BrainTrader(capital=args.capital, paper_trading=args.paper)
    portfolio = Portfolio(initial_capital=args.capital)
    
    # Boucle principale
    cycle = 0
    
    try:
        while True:
            cycle += 1
            
            if args.cycles and cycle > args.cycles:
                logger.info(f"\n✅ {args.cycles} cycles terminés")
                break
            
            logger.info(f"\n📍 Cycle {cycle}/{args.cycles or '∞'}")
            
            # Exécuter cycle
            run_trading_cycle(trader, portfolio)
            
            # Attendre
            if args.cycles is None or cycle < args.cycles:
                logger.info(f"\n⏳ Prochain cycle dans {args.interval} minutes...")
                time.sleep(args.interval * 60)
    
    except KeyboardInterrupt:
        logger.info("\n\n🛑 Arrêt manuel")
    
    except Exception as e:
        logger.error(f"\n❌ Erreur: {e}", exc_info=True)
    
    finally:
        # Résumé final
        logger.info("\n" + "="*70)
        logger.info("📊 RÉSUMÉ FINAL")
        logger.info("="*70)
        
        summary = portfolio.get_summary()
        logger.info(f"💰 Capital initial: ${portfolio.initial_capital:,.2f}")
        logger.info(f"💵 Valeur finale: ${summary['total_value']:,.2f}")
        logger.info(f"📈 P&L total: ${summary['total_return']:+,.2f} ({summary['total_return_pct']:+.2f}%)")
        logger.info(f"🔢 Trades exécutés: {summary['trades_count']}")
        
        portfolio.save_state('portfolio_final.json')

if __name__ == "__main__":
    main()
