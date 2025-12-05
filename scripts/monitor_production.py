#!/usr/bin/env python3
"""
Script de Monitoring Production pour Ploutos
Vérifie dérive modèle et lance alertes si nécessaire

Usage:
    python3 scripts/monitor_production.py --model models/stage1_final.zip
    python3 scripts/monitor_production.py --model models/stage1_final.zip --auto-retrain
"""

import sys
sys.path.insert(0, '.')

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from stable_baselines3 import PPO

from core.drift_detector import ModelDriftDetector
from core.data_fetcher import UniversalDataFetcher
from core.universal_environment import UniversalTradingEnv

def load_baseline_data(cache_file='data_cache/baseline_stats.csv'):
    """
    Charge données baseline (train/val)
    """
    
    if os.path.exists(cache_file):
        print(f"✅ Chargement baseline depuis cache : {cache_file}")
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)
    
    # Si pas de cache, utiliser données récentes
    print("⚠️  Pas de baseline trouvée, utilisation données récentes")
    
    fetcher = UniversalDataFetcher()
    
    # ✅ FIX : Utiliser start_date/end_date au lieu de period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 ans
    
    data = fetcher.fetch(
        'SPY',
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        interval='1h'
    )
    
    # Prendre 80% comme baseline
    baseline = data.iloc[:int(len(data)*0.8)]
    
    # Sauvegarder pour prochaine fois
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    baseline.to_csv(cache_file)
    
    print(f"✅ Baseline créée : {len(baseline)} lignes sauvegardées")
    
    return baseline

def fetch_production_data(days=30):
    """
    Récupère données production (derniers X jours)
    
    Args:
        days: Nombre de jours à récupérer
    """
    
    print(f"\n📥 Récupération données production ({days} derniers jours)...")
    
    fetcher = UniversalDataFetcher()
    
    # ✅ FIX : Utiliser start_date/end_date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = fetcher.fetch(
        'SPY',
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        interval='1h'
    )
    
    print(f"✅ {len(data)} bougies récupérées")
    
    return data

def calculate_current_performance(model, data, episodes=5):
    """
    Évalue performance actuelle du modèle
    """
    
    print("\n📊 Évaluation performance actuelle...")
    
    returns = []
    
    for ep in range(episodes):
        env = UniversalTradingEnv(
            data={'SPY': data},
            initial_balance=10000,
            commission=0.001,
            max_steps=min(500, len(data) - 110)
        )
        
        obs, _ = env.reset()
        done = False
        episode_values = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_values.append(info['portfolio_value'])
        
        # Calculer return
        if len(episode_values) > 10:
            df_val = pd.DataFrame({'value': episode_values})
            df_val['ret'] = df_val['value'].pct_change().fillna(0)
            
            episode_return = (episode_values[-1] - episode_values[0]) / episode_values[0]
            returns.append(episode_return)
    
    # Métriques
    if len(returns) == 0:
        return {'sharpe': 0, 'mean_return': 0, 'max_dd': 0}
    
    returns_series = pd.Series(returns)
    
    sharpe = (returns_series.mean() / returns_series.std()) * np.sqrt(252) if returns_series.std() > 0 else 0
    mean_return = returns_series.mean() * 100
    
    # Max DD approximatif
    cumulative = (1 + returns_series).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min() * 100
    
    performance = {
        'sharpe': float(sharpe),
        'mean_return': float(mean_return),
        'max_dd': float(max_dd)
    }
    
    print(f"  Sharpe Ratio : {sharpe:.2f}")
    print(f"  Mean Return  : {mean_return:.2f}%")
    print(f"  Max DD       : {max_dd:.2f}%")
    
    return performance

def monitor(model_path, auto_retrain=False, sensitivity='medium', production_days=30):
    """
    Lance monitoring complet
    
    Args:
        model_path: Chemin vers modèle .zip
        auto_retrain: Si True, retraîne automatiquement si drift
        sensitivity: 'low'|'medium'|'high'
        production_days: Nombre de jours de données production à analyser
    """
    
    print("\n" + "="*80)
    print("🔍 MONITORING PRODUCTION PLOUTOS")
    print("="*80)
    print(f"  Modèle       : {model_path}")
    print(f"  Auto-retrain : {'Activé' if auto_retrain else 'Désactivé'}")
    print(f"  Sensibilité  : {sensitivity}")
    print(f"  Date         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Charger modèle
    if not os.path.exists(model_path):
        print(f"❌ Modèle introuvable : {model_path}")
        return
    
    print(f"🧠 Chargement modèle...")
    model = PPO.load(model_path)
    print(f"✅ Modèle chargé")
    
    # 2. Charger baseline
    baseline_data = load_baseline_data()
    
    baseline_performance = {
        'sharpe': 1.5,  # À remplacer par vraies valeurs baseline
        'max_dd': -12.0
    }
    
    # 3. Initialiser detector
    detector = ModelDriftDetector(
        baseline_data=baseline_data,
        baseline_performance=baseline_performance,
        sensitivity=sensitivity
    )
    
    # 4. Récupérer données production
    production_data = fetch_production_data(days=production_days)
    
    # 5. Évaluer performance actuelle
    current_performance = calculate_current_performance(model, production_data)
    
    # 6. Détecter dérive
    print("\n" + "="*80)
    print("🔎 DÉTECTION DE DÉRIVE")
    print("="*80)
    
    drift_result = detector.detect_drift(
        new_data=production_data,
        new_performance=current_performance
    )
    
    # 7. Résultats
    print("\n" + "="*80)
    print("📋 RÉSULTATS")
    print("="*80)
    
    if drift_result['drift_detected']:
        print(f"❌ Dérive détectée : {drift_result['drift_type'].upper()}")
        print(f"   Sévérité : {drift_result['severity'].upper()}")
        
        print("\n📌 Détails :")
        if 'drifted_features' in drift_result['details']:
            print(f"   Features dérivées : {len(drift_result['details']['drifted_features'])}")
            for feat in drift_result['details']['drifted_features'][:5]:
                psi = drift_result['metrics']['data_drift']['psi_scores'].get(feat, 0)
                print(f"     - {feat} (PSI: {psi:.3f})")
        
        print("\n🚨 Actions recommandées :")
        for rec in drift_result['recommendations']:
            print(f"   {rec}")
        
        # Auto-retrain si activé
        if auto_retrain and drift_result['severity'] in ['medium', 'high']:
            print("\n🔄 AUTO-RETRAIN ACTIVÉ")
            print("   ⚠️  Fonctionnalité à implémenter :")
            print("   - Recharger données récentes (3 derniers mois)")
            print("   - Lancer train_curriculum.py --stage 1")
            print("   - Remplacer modèle actuel après validation")
        
    else:
        print("✅ Aucune dérive détectée")
        print("   Le modèle fonctionne correctement")
    
    # 8. Export rapport
    detector.export_report('reports/drift_monitoring_latest.json')
    
    # 9. Résumé final
    summary = detector.get_drift_summary()
    
    print("\n" + "="*80)
    print("📊 RÉSUMÉ SESSION")
    print("="*80)
    print(f"  Checks total   : {summary['total_checks']}")
    print(f"  Drift events   : {summary['drift_events']}")
    print(f"  Drift rate     : {summary['drift_rate']:.1f}%")
    
    if summary['drift_by_type']:
        print("\n  Par type :")
        for dtype, count in summary['drift_by_type'].items():
            print(f"    - {dtype.upper()} : {count}")
    
    print("\n" + "="*80)
    print(f"✅ Monitoring terminé - {datetime.now().strftime('%H:%M:%S')}")
    print("="*80 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Monitoring Production Ploutos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Monitoring simple
  python3 scripts/monitor_production.py --model models/stage1_final.zip
  
  # Monitoring avec auto-retrain
  python3 scripts/monitor_production.py --model models/stage1_final.zip --auto-retrain
  
  # Haute sensibilité
  python3 scripts/monitor_production.py --model models/stage1_final.zip --sensitivity high
  
  # Analyser 60 derniers jours
  python3 scripts/monitor_production.py --model models/stage1_final.zip --days 60
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                        help='Chemin vers modèle .zip')
    parser.add_argument('--auto-retrain', action='store_true',
                        help='Active retraînement automatique si drift')
    parser.add_argument('--sensitivity', type=str, default='medium',
                        choices=['low', 'medium', 'high'],
                        help='Sensibilité détection drift')
    parser.add_argument('--days', type=int, default=30,
                        help='Nombre de jours de données production à analyser (défaut: 30)')
    
    args = parser.parse_args()
    
    monitor(
        model_path=args.model,
        auto_retrain=args.auto_retrain,
        sensitivity=args.sensitivity,
        production_days=args.days
    )
