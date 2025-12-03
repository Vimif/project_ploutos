#!/usr/bin/env python3
# scripts/run_live_trader.py
"""Lancer le live trader avec Alpaca"""

# === FIX PATH ===
import sys
from pathlib import Path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))
# ================

import argparse
from trading.live_trader import LiveTrader

def main():
    parser = argparse.ArgumentParser(description='Live Trader avec Alpaca')
    
    parser.add_argument(
        '--live',
        action='store_true',
        help='🔴 MODE LIVE (trades réels)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Intervalle entre analyses (minutes)'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=None,
        help='Capital initial (optionnel, sinon utilise le compte Alpaca)'
    )
    
    args = parser.parse_args()
    
    # Initialiser et lancer
    trader = LiveTrader(
        paper_trading=not args.live,
        capital=args.capital
    )
    
    trader.run(check_interval_minutes=args.interval)

if __name__ == "__main__":
    main()
