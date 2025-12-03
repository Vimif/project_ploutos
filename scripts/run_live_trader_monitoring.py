#!/usr/bin/env python3
"""Lancer LiveTrader avec monitoring intégré"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from trading.live_trader import LiveTrader

def main():
    parser = argparse.ArgumentParser(description='Ploutos Live Trader with Monitoring')
    parser.add_argument('--paper', action='store_true', help='Paper trading mode')
    parser.add_argument('--interval', type=int, default=60, help='Check interval (minutes)')
    parser.add_argument('--capital', type=float, help='Initial capital')
    parser.add_argument('--monitoring-port', type=int, default=9090, help='Prometheus port')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 PLOUTOS TRADING BOT - FULL FEATURED")
    print("="*70)
    print(f"Mode: {'PAPER' if args.paper else 'LIVE'}")
    print(f"Interval: {args.interval} minutes")
    print(f"Monitoring: Port {args.monitoring_port}")
    print("="*70)
    
    trader = LiveTrader(
        paper_trading=args.paper,
        capital=args.capital,
        monitoring_port=args.monitoring_port
    )
    
    trader.run(check_interval_minutes=args.interval)

if __name__ == '__main__':
    main()