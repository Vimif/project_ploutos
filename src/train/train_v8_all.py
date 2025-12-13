#!/usr/bin/env python3
"""
🚀 PLOUTOS V8 - ENTRAINEMENT COMPLET

Entraîne tous les modèles V8 Oracle en une seule commande

Usage:
    python src/train/train_v8_all.py
    python src/train/train_v8_all.py --quick  # Entraînement rapide
    python src/train/train_v8_all.py --models lightgbm,xgboost  # Modèles spécifiques

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from datetime import datetime, timedelta

try:
    from src.models.v8_lightgbm_intraday import V8LightGBMIntraday
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False

try:
    from src.models.v8_xgboost_weekly import V8XGBoostWeekly
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(description='Entraînement V8 Oracle')
    parser.add_argument('--models', type=str, default='all',
                       help='Modèles à entraîner (lightgbm,xgboost,all)')
    parser.add_argument('--quick', action='store_true',
                       help='Entraînement rapide (moins de données)')
    parser.add_argument('--start-date', type=str, default='2023-01-01',
                       help='Date de début (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-12-01',
                       help='Date de fin (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Tickers
    if args.quick:
        tickers = ['NVDA', 'MSFT', 'AAPL', 'SPY']
    else:
        tickers = ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'TSLA', 
                   'JPM', 'BAC', 'XOM', 'SPY', 'QQQ']
    
    models_to_train = args.models.lower().split(',')
    
    print("\n" + "="*70)
    print("🚀 PLOUTOS V8 ORACLE - ENTRAINEMENT COMPLET")
    print("="*70)
    print(f"\n📅 Période: {args.start_date} à {args.end_date}")
    print(f"🎯 Tickers: {', '.join(tickers)}")
    print(f"🧪 Modèles: {args.models}")
    print(f"⚡ Mode: {'QUICK' if args.quick else 'FULL'}\n")
    
    results = {}
    
    # LightGBM Intraday (1 jour)
    if ('all' in models_to_train or 'lightgbm' in models_to_train) and LIGHTGBM_AVAILABLE:
        print("\n" + "#"*70)
        print("# 1. LIGHTGBM INTRADAY (1 JOUR)")
        print("#"*70 + "\n")
        
        predictor = V8LightGBMIntraday()
        res = predictor.train(
            tickers=tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            test_size=0.2
        )
        predictor.save('models/v8_lightgbm_intraday.pkl')
        results['lightgbm_intraday'] = res
    
    # XGBoost Weekly (5 jours)
    if ('all' in models_to_train or 'xgboost' in models_to_train) and XGBOOST_AVAILABLE:
        print("\n" + "#"*70)
        print("# 2. XGBOOST WEEKLY (5 JOURS)")
        print("#"*70 + "\n")
        
        predictor = V8XGBoostWeekly()
        res = predictor.train(
            tickers=tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            test_size=0.2
        )
        predictor.save('models/v8_xgboost_weekly.pkl')
        results['xgboost_weekly'] = res
    
    # Résumé final
    print("\n" + "="*70)
    print("✅ ENTRAINEMENT TERMINÉ")
    print("="*70)
    
    print("\n📊 RÉSUMÉ DES PERFORMANCES:\n")
    
    for model_name, res in results.items():
        print(f"{model_name.upper()}:")
        print(f"  Train Accuracy: {res['train_accuracy']*100:.2f}%")
        print(f"  Test Accuracy:  {res['test_accuracy']*100:.2f}%")
        print()
    
    print("💾 Modèles sauvegardés dans models/")
    print("\n✅ Prêt pour la production !\n")


if __name__ == '__main__':
    main()
