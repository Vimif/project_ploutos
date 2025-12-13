#!/usr/bin/env python3
"""
⚡ PLOUTOS QUICK TEST

Script de test rapide pour comparer les performances des modèles
sans deployer en production

Usage:
    python tests/quick_test.py --days 30 --tickers NVDA,MSFT,AAPL
    python tests/quick_test.py --preset tech
    python tests/quick_test.py --full  # Test complet

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from backtest_framework import BacktestFramework

# Presets de tickers
TICKER_PRESETS = {
    'tech': ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'TSLA'],
    'finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK'],
    'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC'],
    'defensive': ['SPY', 'QQQ', 'VOO', 'VTI', 'IWM', 'DIA', 'VEA'],
    'mixed': ['NVDA', 'MSFT', 'JPM', 'XOM', 'SPY', 'QQQ', 'AAPL'],
    'full': ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'JPM', 'SPY', 'QQQ', 'XOM', 'CVX']
}

def main():
    parser = argparse.ArgumentParser(description='Quick test des modèles Ploutos')
    
    parser.add_argument('--preset', type=str, choices=list(TICKER_PRESETS.keys()),
                       help='Preset de tickers (tech, finance, energy, defensive, mixed, full)')
    parser.add_argument('--tickers', type=str,
                       help='Liste de tickers (séparés par virgules)')
    parser.add_argument('--days', type=int, default=90,
                       help='Nombre de jours pour le backtest (défaut: 90)')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Capital initial (défaut: 100000)')
    parser.add_argument('--model', type=str, default='models/autonomous/production.zip',
                       help='Chemin vers le modèle PPO')
    parser.add_argument('--test-split', type=float, default=0.3,
                       help='Proportion de données pour le test (défaut: 0.3)')
    parser.add_argument('--skip-v7', action='store_true',
                       help='Ne pas tester V7')
    parser.add_argument('--skip-hybrid', action='store_true',
                       help='Ne pas tester le système hybride')
    
    args = parser.parse_args()
    
    # Déterminer les tickers
    if args.preset:
        tickers = TICKER_PRESETS[args.preset]
        print(f"✅ Preset '{args.preset}' sélectionné")
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
    else:
        tickers = TICKER_PRESETS['mixed']
        print("ℹ️  Aucun ticker spécifié - utilise preset 'mixed'")
    
    print(f"\n🎯 Tickers: {', '.join(tickers)}")
    print(f"📅 Période: {args.days} derniers jours")
    print(f"💰 Capital: ${args.capital:,.0f}")
    
    # Calculer dates
    from datetime import datetime, timedelta
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=args.days + 200)).strftime('%Y-%m-%d')
    
    # Initialiser framework
    framework = BacktestFramework(
        initial_capital=args.capital,
        commission=0.001,
        slippage=0.0005
    )
    
    # Charger données
    data = framework.load_historical_data(tickers, start_date, end_date)
    
    if len(data) == 0:
        print("❌ Aucune donnée chargée")
        return
    
    # Tests
    print("\n" + "="*70)
    print("🚦 DÉMARRAGE DES TESTS")
    print("="*70)
    
    # Test PPO only
    print("\n🔵 Test 1/3: PPO Only...")
    try:
        framework.backtest_ppo_only(
            model_path=args.model,
            data=data,
            test_split=args.test_split
        )
    except Exception as e:
        print(f"❌ Erreur PPO: {e}")
    
    # Test V7 only
    if not args.skip_v7:
        print("\n🟡 Test 2/3: V7 Enhanced Only...")
        try:
            framework.backtest_v7_only(
                data=data,
                test_split=args.test_split
            )
        except Exception as e:
            print(f"❌ Erreur V7: {e}")
    
    # Test Hybrid
    if not args.skip_hybrid:
        print("\n🟠 Test 3/3: PPO + V7 Hybrid...")
        try:
            framework.backtest_ppo_plus_v7(
                model_path=args.model,
                data=data,
                test_split=args.test_split
            )
        except Exception as e:
            print(f"❌ Erreur Hybrid: {e}")
    
    # Comparaison
    if len(framework.results) > 0:
        print("\n" + "="*70)
        comparison_df = framework.compare_models()
        print("="*70)
        
        # Sauvegarder
        framework.save_results()
    else:
        print("\n❌ Aucun résultat à comparer")
    
    print("\n✅ Tests terminés !\n")

if __name__ == '__main__':
    main()
