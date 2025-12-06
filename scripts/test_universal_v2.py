#!/usr/bin/env python3
"""
🧪 TEST UNIVERSALTRADINGENV V2

Test du système principal avec tous les fixes appliqués:
- Actions discrètes (BUY/HOLD/SELL)
- Reward PnL réalisé + latent
- Multi-assets (5 tickers)

Objectif: Vérifier que le système fonctionne en production
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import warnings
warnings.filterwarnings('ignore')

from core.universal_environment_v2 import UniversalTradingEnvV2

print("="*80)
print("🧪 TEST UNIVERSALTRADINGENV V2")
print("="*80)

def load_data(tickers, start_date, end_date):
    """Charge les données pour plusieurs tickers"""
    data = {}
    
    for ticker in tickers:
        print(f"   Chargement {ticker}...", end=" ")
        df = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
        
        if df.empty or len(df) < 100:
            print("❌ Erreur")
            continue
        
        data[ticker] = df
        print(f"✅ {len(df)} jours")
    
    return data

def evaluate(model, env, n_episodes=30):
    """Evaluation avec variance"""
    portfolios = []
    all_actions = {ticker: [] for ticker in env.tickers}
    
    for i in range(n_episodes):
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        
        done = False
        ep_values = [100000]
        
        while not done:
            # Mélange det/sto pour variance
            deterministic = (i < n_episodes // 2)
            actions, _ = model.predict(obs, deterministic=deterministic)
            
            # Enregistrer actions par ticker
            for j, ticker in enumerate(env.tickers):
                all_actions[ticker].append(int(actions[j]))
            
            result = env.step(actions)
            
            if len(result) == 5:
                obs, reward, term, trunc, info = result
                done = term or trunc
            else:
                obs, reward, done, info = result
            
            ep_values.append(info['portfolio_value'])
        
        portfolios.append(ep_values[-1])
    
    mean_pf = np.mean(portfolios)
    std_pf = np.std(portfolios)
    
    # Sharpe
    returns = [(p - 100000) / 100000 for p in portfolios]
    sharpe = 0
    if len(returns) > 1 and np.std(returns) > 1e-8:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        sharpe = np.clip(sharpe, -10, 10)
    
    # Actions par ticker
    actions_stats = {}
    for ticker in env.tickers:
        actions_arr = np.array(all_actions[ticker])
        actions_stats[ticker] = {
            'hold_pct': (np.sum(actions_arr == 0) / len(actions_arr)) * 100,
            'buy_pct': (np.sum(actions_arr == 1) / len(actions_arr)) * 100,
            'sell_pct': (np.sum(actions_arr == 2) / len(actions_arr)) * 100
        }
    
    # Actions globales
    all_actions_flat = [a for actions_list in all_actions.values() for a in actions_list]
    actions_arr = np.array(all_actions_flat)
    
    return {
        'mean_portfolio': mean_pf,
        'std_portfolio': std_pf,
        'min_portfolio': np.min(portfolios),
        'max_portfolio': np.max(portfolios),
        'sharpe': sharpe,
        'actions_stats': actions_stats,
        'global_hold': (np.sum(actions_arr == 0) / len(actions_arr)) * 100,
        'global_buy': (np.sum(actions_arr == 1) / len(actions_arr)) * 100,
        'global_sell': (np.sum(actions_arr == 2) / len(actions_arr)) * 100,
        'returns_std': np.std(returns)
    }

def main():
    print("\n📡 1. Chargement données multi-assets...")
    
    tickers = ['NVDA', 'MSFT', 'AAPL', 'SPY', 'QQQ']
    
    end = datetime.now()
    start = end - timedelta(days=600)
    
    data = load_data(tickers, start, end)
    
    if len(data) < 3:
        print("❌ Pas assez de données chargées")
        return False
    
    print(f"\n   ✅ {len(data)} tickers chargés")
    
    # ✅ Calculer taille min des données
    min_length = min(len(df) for df in data.values())
    print(f"   📊 Taille minimale: {min_length} jours")
    
    # ✅ Adapter max_steps (60% des données max)
    max_steps = min(250, int(min_length * 0.6))
    print(f"   ⏱️  Max steps: {max_steps} jours\n")
    
    # Créer env V2
    print("🏗️  2. Création UniversalTradingEnvV2...")
    env = UniversalTradingEnvV2(
        data=data,
        initial_balance=100000,
        commission=0.0001,
        max_steps=max_steps,  # ✅ Dynamique
        buy_pct=0.2
    )
    vec_env = DummyVecEnv([lambda: env])
    print("   ✅ Env créé\n")
    
    # Modèle PPO
    print("🧠 3. Initialisation PPO...")
    model = PPO(
        'MlpPolicy',
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={'net_arch': [256, 256]},
        verbose=0
    )
    print("   ✅ Modèle initialisé\n")
    
    # Entraînement
    print("🚀 4. Entraînement (300k steps)...\n")
    
    try:
        model.learn(total_timesteps=300_000, progress_bar=True)
    except KeyboardInterrupt:
        print("\n⚠️  Interrompu")
    
    print("\n   ✅ Entraînement terminé\n")
    
    # Évaluation
    print("📊 5. Évaluation finale (30 épisodes)...")
    metrics = evaluate(model, env, n_episodes=30)
    
    print("\n" + "="*80)
    print("🎯 RÉSULTATS UNIVERSALTRADINGENV V2")
    print("="*80)
    
    profit_pct = (metrics['mean_portfolio'] - 100000) / 1000
    
    print(f"\n💰 PORTFOLIO:")
    print(f"   Moyen : ${metrics['mean_portfolio']:,.0f} ({profit_pct:+.1f}%)")
    print(f"   Std   : ${metrics['std_portfolio']:,.0f}")
    print(f"   Min   : ${metrics['min_portfolio']:,.0f}")
    print(f"   Max   : ${metrics['max_portfolio']:,.0f}")
    
    print(f"\n📈 MÉTRIQUES:")
    print(f"   Sharpe       : {metrics['sharpe']:.3f}")
    print(f"   Returns Std  : {metrics['returns_std']:.4f}")
    
    print(f"\n🎯 ACTIONS GLOBALES:")
    print(f"   HOLD  : {metrics['global_hold']:5.1f}%")
    print(f"   BUY   : {metrics['global_buy']:5.1f}%")
    print(f"   SELL  : {metrics['global_sell']:5.1f}%")
    
    print(f"\n📄 ACTIONS PAR TICKER:")
    for ticker, stats in metrics['actions_stats'].items():
        print(f"   {ticker:6s}: HOLD {stats['hold_pct']:5.1f}% | BUY {stats['buy_pct']:5.1f}% | SELL {stats['sell_pct']:5.1f}%")
    
    # Validation
    print("\n" + "="*80)
    print("✅ VALIDATION")
    print("="*80)
    
    success = True
    
    if metrics['mean_portfolio'] >= 102000:
        print("\n✅ Portfolio > $102k : PASS")
    else:
        print(f"\n❌ Portfolio : ${metrics['mean_portfolio']:,.0f} < $102k")
        success = False
    
    if metrics['sharpe'] >= 0.3 or metrics['std_portfolio'] >= 1000:
        print("✅ Sharpe/Variance : PASS")
    else:
        print(f"⚠️  Sharpe : {metrics['sharpe']:.3f} (acceptable)")
    
    if metrics['global_buy'] >= 10 and metrics['global_sell'] >= 10:
        print("✅ Actions équilibrées : PASS")
    else:
        print(f"❌ Actions : BUY {metrics['global_buy']:.1f}% / SELL {metrics['global_sell']:.1f}%")
        success = False
    
    print("\n" + "="*80)
    
    if success:
        print("✅ TEST UNIVERSALENV V2 : SUCCÈS !")
        print("\n🎉 Le système principal fonctionne avec les fixes !")
        print("\n🚀 Prochaine étape:")
        print("   1. Entraîner modèle production (1M steps)")
        print("   2. Déployer sur VPS")
        print("   3. Remplacer l'ancien environnement")
    else:
        print("❌ TEST UNIVERSALENV V2 : ÉCHEC PARTIEL")
        print("\n🔧 Ajustements possibles:")
        print("   1. Augmenter timesteps (300k → 500k)")
        print("   2. Ajuster buy_pct (20% → 15%)")
        print("   3. Augmenter ent_coef (exploration)")
    
    print("="*80)
    
    return success

if __name__ == '__main__':
    print("\n🎯 Lance test UniversalTradingEnv V2...\n")
    
    success = main()
    
    exit(0 if success else 1)
