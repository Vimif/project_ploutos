#!/usr/bin/env python3
"""🔍 DIAGNOSTIC COMPORTEMENT MODÈLE

Analyse pourquoi le modèle ne trade pas

Ce script:
1. Charge le modèle
2. Analyse la distribution des actions
3. Vérifie les observations
4. Teste avec/sans mode deterministic
5. Génère recommandations

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from core.data_fetcher import UniversalDataFetcher
from core.universal_environment_v4_ultimate import UniversalTradingEnvV4Ultimate

# Config
TICKERS = [
    'NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'TSLA',
    'SPY', 'QQQ', 'VOO', 'VTI',
    'XLE', 'XLF', 'XLK', 'XLV'
]

INITIAL_BALANCE = 100000


class ModelDiagnostic:
    """Diagnostic complet du modèle"""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        self.env = None
        self.results = defaultdict(list)
        
    def load_model(self):
        """Charger le modèle"""
        print(f"\n📦 Chargement modèle: {self.model_path.name}")
        
        if not self.model_path.exists():
            print(f"❌ Modèle introuvable")
            return False
        
        self.model = PPO.load(self.model_path)
        print(f"✅ Modèle chargé")
        print(f"  • Observation space: {self.model.observation_space.shape}")
        print(f"  • Action space: {self.model.action_space}")
        
        return True
    
    def load_data(self, days=30):
        """Charger données"""
        print(f"\n📊 Chargement données ({days} jours)...")
        
        fetcher = UniversalDataFetcher()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 50)
        
        data = {}
        for ticker in TICKERS:
            try:
                df = fetcher.fetch(
                    ticker,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    interval='1h'
                )
                if df is not None and len(df) > 50:
                    data[ticker] = df
                    print(f"  ✅ {ticker}: {len(df)} barres")
            except Exception as e:
                print(f"  ❌ {ticker}: {e}")
        
        print(f"\n✅ {len(data)} tickers chargés")
        return data
    
    def create_environment(self, data):
        """Créer environnement"""
        print(f"\n🏭 Création environnement...")
        
        self.env = UniversalTradingEnvV4Ultimate(
            data=data,
            initial_balance=INITIAL_BALANCE,
            commission=0.0,
            sec_fee=0.0000221,
            finra_taf=0.000145,
            max_steps=1000,
            buy_pct=0.2,
            slippage_model='realistic',
            spread_bps=2.0,
            max_position_pct=0.25,
            max_trades_per_day=3,
            min_holding_period=0
        )
        
        print(f"✅ Environnement créé")
        return self.env
    
    def test_action_distribution(self, steps=500, deterministic=True):
        """✅ Analyser distribution des actions"""
        mode = "DETERMINISTIC" if deterministic else "STOCHASTIC"
        print(f"\n🎲 Test distribution actions ({mode})...")
        print(f"  • Steps: {steps}")
        
        obs, _ = self.env.reset()
        
        action_counts = Counter()
        action_history = []
        obs_stats = []
        
        for step in range(steps):
            # Prédiction
            action, _states = self.model.predict(obs, deterministic=deterministic)
            
            # Stats observations
            obs_stats.append({
                'min': obs.min(),
                'max': obs.max(),
                'mean': obs.mean(),
                'std': obs.std(),
                'nan_count': np.isnan(obs).sum(),
                'inf_count': np.isinf(obs).sum()
            })
            
            # Enregistrer actions
            for i, a in enumerate(action):
                action_key = f"Ticker_{i}_Action_{a}"
                action_counts[action_key] += 1
            
            action_history.append(action.copy())
            
            # Step
            obs, reward, done, truncated, info = self.env.step(action)
            
            if done or truncated:
                obs, _ = self.env.reset()
        
        # Analyser par action (0=HOLD, 1=BUY, 2=SELL)
        action_history = np.array(action_history)
        
        print(f"\n📊 Distribution globale:")
        total_actions = steps * len(TICKERS)
        
        hold_count = np.sum(action_history == 0)
        buy_count = np.sum(action_history == 1)
        sell_count = np.sum(action_history == 2)
        
        print(f"  • HOLD (0): {hold_count:,} ({hold_count/total_actions*100:.1f}%)")
        print(f"  • BUY  (1): {buy_count:,} ({buy_count/total_actions*100:.1f}%)")
        print(f"  • SELL (2): {sell_count:,} ({sell_count/total_actions*100:.1f}%)")
        
        # Par ticker
        print(f"\n📊 Distribution par ticker:")
        for i, ticker in enumerate(TICKERS):
            ticker_actions = action_history[:, i]
            hold = np.sum(ticker_actions == 0)
            buy = np.sum(ticker_actions == 1)
            sell = np.sum(ticker_actions == 2)
            
            print(f"  {ticker:5s}: HOLD={hold:3d} ({hold/steps*100:5.1f}%) | "
                  f"BUY={buy:3d} ({buy/steps*100:5.1f}%) | "
                  f"SELL={sell:3d} ({sell/steps*100:5.1f}%)")
        
        # Stats observations
        print(f"\n🔍 Stats observations:")
        obs_df = pd.DataFrame(obs_stats)
        print(f"  • Min range: [{obs_df['min'].min():.2f}, {obs_df['min'].max():.2f}]")
        print(f"  • Max range: [{obs_df['max'].min():.2f}, {obs_df['max'].max():.2f}]")
        print(f"  • Mean: {obs_df['mean'].mean():.2f} ± {obs_df['mean'].std():.2f}")
        print(f"  • Std: {obs_df['std'].mean():.2f}")
        print(f"  • NaN count: {obs_df['nan_count'].sum()}")
        print(f"  • Inf count: {obs_df['inf_count'].sum()}")
        
        return {
            'hold_pct': hold_count / total_actions,
            'buy_pct': buy_count / total_actions,
            'sell_pct': sell_count / total_actions,
            'action_history': action_history,
            'obs_stats': obs_df
        }
    
    def compare_deterministic_modes(self, steps=200):
        """✅ Comparer mode deterministic vs stochastic"""
        print(f"\n🎯 Comparaison modes...")
        
        print(f"\n1️⃣  Mode DETERMINISTIC (inference):")
        det_results = self.test_action_distribution(steps=steps, deterministic=True)
        
        print(f"\n2️⃣  Mode STOCHASTIC (exploration):")
        stoch_results = self.test_action_distribution(steps=steps, deterministic=False)
        
        # Comparaison
        print(f"\n🔄 COMPARAISON:")
        print(f"  {'Mode':15s} | {'HOLD':>8s} | {'BUY':>8s} | {'SELL':>8s}")
        print(f"  {'-'*15:15s} | {'-'*8:8s} | {'-'*8:8s} | {'-'*8:8s}")
        print(f"  {'Deterministic':15s} | {det_results['hold_pct']*100:7.1f}% | "
              f"{det_results['buy_pct']*100:7.1f}% | {det_results['sell_pct']*100:7.1f}%")
        print(f"  {'Stochastic':15s} | {stoch_results['hold_pct']*100:7.1f}% | "
              f"{stoch_results['buy_pct']*100:7.1f}% | {stoch_results['sell_pct']*100:7.1f}%")
        
        return det_results, stoch_results
    
    def test_policy_bias(self):
        """✅ Tester si policy est biasée vers HOLD"""
        print(f"\n🧪 Test bias policy...")
        
        # Tester avec observations artificielles
        test_cases = [
            ('Zeros', np.zeros(self.model.observation_space.shape)),
            ('Ones', np.ones(self.model.observation_space.shape)),
            ('Random', np.random.randn(*self.model.observation_space.shape)),
            ('Extreme Positive', np.ones(self.model.observation_space.shape) * 5),
            ('Extreme Negative', np.ones(self.model.observation_space.shape) * -5),
        ]
        
        print(f"\n  Test avec observations artificielles:")
        for name, obs in test_cases:
            action, _ = self.model.predict(obs, deterministic=True)
            
            hold = np.sum(action == 0)
            buy = np.sum(action == 1)
            sell = np.sum(action == 2)
            total = len(action)
            
            print(f"    {name:20s}: HOLD={hold:2d}/{total} ({hold/total*100:5.1f}%) | "
                  f"BUY={buy:2d} | SELL={sell:2d}")
    
    def generate_recommendations(self, results):
        """✅ Générer recommandations"""
        print(f"\n\n{'='*70}")
        print(f"💡 RECOMMANDATIONS")
        print(f"{'='*70}\n")
        
        hold_pct = results['hold_pct']
        buy_pct = results['buy_pct']
        sell_pct = results['sell_pct']
        
        if hold_pct > 0.9:
            print("❌ PROBLÈME MAJEUR: Modèle fait 90%+ de HOLD")
            print("\n  Causes possibles:")
            print("    1. ⚠️  Entraînement insuffisant (pas assez de timesteps)")
            print("    2. ⚠️  Reward shaping inadapté (pénalités trop fortes)")
            print("    3. ⚠️  Contraintes trop strictes (PDT, holding period)")
            print("    4. ⚠️  Exploration insuffisante (entropy coef trop bas)")
            
            print("\n  Solutions à essayer:")
            print("    ✅ 1. Augmenter entropy_coef: 0.01 → 0.05")
            print("    ✅ 2. Réduire pénalités (drawdown_penalty, overtrading)")
            print("    ✅ 3. Assouplir contraintes (max_trades_per_day: 3 → 10)")
            print("    ✅ 4. Re-entraîner avec plus de timesteps (10M → 20M)")
            print("    ✅ 5. Reward bonus pour trades réussis")
            
        elif hold_pct > 0.7:
            print("⚠️  ATTENTION: Modèle trop conservateur (70%+ HOLD)")
            print("\n  Solutions:")
            print("    ✅ Augmenter légèrement entropy_coef")
            print("    ✅ Réduire min_holding_period")
            print("    ✅ Augmenter buy_pct (plus de capital par trade)")
            
        else:
            print("✅ Distribution actions semble correcte")
            print(f"  • HOLD: {hold_pct*100:.1f}%")
            print(f"  • BUY:  {buy_pct*100:.1f}%")
            print(f"  • SELL: {sell_pct*100:.1f}%")
        
        print(f"\n{'='*70}\n")
    
    def run_full_diagnostic(self):
        """Exécuter diagnostic complet"""
        print("\n" + "="*70)
        print("🔍 DIAGNOSTIC COMPORTEMENT MODÈLE")
        print("="*70)
        
        # 1. Charger modèle
        if not self.load_model():
            return
        
        # 2. Charger données
        data = self.load_data(days=30)
        if len(data) < 5:
            print("❌ Pas assez de données")
            return
        
        # 3. Créer environnement
        self.create_environment(data)
        
        # 4. Test distribution actions
        det_results, stoch_results = self.compare_deterministic_modes(steps=300)
        
        # 5. Test bias policy
        self.test_policy_bias()
        
        # 6. Recommandations
        self.generate_recommendations(det_results)
        
        print("✅ Diagnostic terminé !\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnostic modèle Ploutos')
    parser.add_argument('--model', type=str, required=True, help='Chemin du modèle')
    args = parser.parse_args()
    
    diagnostic = ModelDiagnostic(model_path=args.model)
    diagnostic.run_full_diagnostic()
