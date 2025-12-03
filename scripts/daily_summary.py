#!/usr/bin/env python3
"""Job quotidien pour sauvegarder le résumé"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import date
from trading.alpaca_client import AlpacaClient
from database.db import save_daily_summary, get_trade_history

def run_daily_summary():
    """Exécuter le résumé quotidien"""
    print(f"📅 Génération du résumé pour {date.today()}")
    
    try:
        # Connexion Alpaca
        client = AlpacaClient(paper_trading=True)
        
        # Récupérer les données
        account = client.get_account()
        positions = client.get_positions()
        
        # Trades du jour
        trades_today = get_trade_history(days=1)
        
        # P&L total
        total_pl = sum(float(p['unrealized_pl']) for p in positions)
        
        # Sauvegarder
        save_daily_summary(
            date=date.today(),
            portfolio_value=float(account['portfolio_value']),
            cash=float(account['cash']),
            buying_power=float(account['buying_power']),
            total_pl=total_pl,
            positions_count=len(positions),
            trades_count=len(trades_today)
        )
        
        # Logger les positions
        client.log_current_positions()
        
        print(f"✅ Résumé sauvegardé:")
        print(f"   Portfolio: ${account['portfolio_value']:,.2f}")
        print(f"   Positions: {len(positions)}")
        print(f"   Trades: {len(trades_today)}")
        print(f"   P&L: ${total_pl:+,.2f}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    run_daily_summary()