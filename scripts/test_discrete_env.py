#!/usr/bin/env python3
"""
🧪 TEST ENVIRONNEMENT ACTIONS DISCRÈTES

Test final avec actions discrètes:
- 0 = HOLD
- 1 = BUY (20% du portfolio)
- 2 = SELL (tout vendre)

Objectif:
- Portfolio > $102k
- Sharpe > 0.5
- Actions équilibrées (BUY > 10%, SELL > 10%)
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

from core.discrete_trading_env import DiscreteTradingEnv

print("="*80)
print("🧪 TEST ENVIRONNEMENT ACTIONS DISCRÈTES")
print("="*80)

def evaluate(model, env, n_episodes=20):
    portfolios = []
    all_actions = []
    
    for _ in range(n_episodes):
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        
        done = False
        ep_values = [100000]
        ep_actions = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            ep_actions.append(int(action))
            
            result = env.step(action)
            
            if len(result) == 5:
                obs, reward, term, trunc, info = result
                done = term or trunc
            else:
                obs, reward, done, info = result
            
            ep_values.append(info['portfolio_value'])
        
        portfolios.append(ep_values[-1])
        all_actions.extend(ep_actions)
    
    mean_pf = np.mean(portfolios)
    
    # Sharpe
    returns = [(p - 100000) / 100000 for p in portfolios]
    sharpe = 0
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        sharpe = np.clip(sharpe, -10, 10)
    
    # Actions
    actions_arr = np.array(all_actions)
    hold_pct = (np.sum(actions_arr == 0) / len(actions_arr)) * 100
    buy_pct = (np.sum(actions_arr == 1) / len(actions_arr)) * 100
    sell_pct = (np.sum(actions_arr == 2) / len(actions_arr)) * 100
    
    return {
        'mean_portfolio': mean_pf,
        'min_portfolio': np.min(portfolios),
        'max_portfolio': np.max(portfolios),
        'sharpe': sharpe,
        'hold_pct': hold_pct,
        'buy_pct': buy_pct,
        'sell_pct': sell_pct
    }

def main():
    print("\n📡 1. Chargement données SPY...")
    
    end = datetime.now()
    start = end - timedelta(days=600)
    
    df = yf.download('SPY', start=start, end=end, interval='1d', progress=False)
    df = df.tail(min(500, len(df)))
    prices = df['Close'].values
    
    print(f"   ✅ {len(prices)} jours chargés\n")
    
    # Créer env avec actions DISCRÈTES
    print("🏗️  2. Création env discret...")
    env = DiscreteTradingEnv(prices, initial_balance=100000, commission=0.0001, max_steps=400, buy_pct=0.2)
    vec_env = DummyVecEnv([lambda: env])
    print("   ✅ Env créé (Action space: Discrete(3))\n")
    
    # Modèle PPO
    print("🧠 3. Initialisation PPO...")
    model = PPO(
        'MlpPolicy',
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,  # ✅ Augmenté pour encourager exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={'net_arch': [128, 128]},
        verbose=0
    )
    print("   ✅ Modèle initialisé\n")
    
    # Entraînement
    print("🚀 4. Entraînement (200k steps)...\n")
    
    try:
        model.learn(total_timesteps=200_000, progress_bar=True)
    except KeyboardInterrupt:
        print("\n⚠️  Interrompu")
    
    print("\n   ✅ Entraînement terminé\n")
    
    # Évaluation
    print("📊 5. Évaluation finale...")
    metrics = evaluate(model, env, n_episodes=20)
    
    print("\n" + "="*80)
    print("🎯 RÉSULTATS TEST DISCRET")
    print("="*80)
    
    profit_pct = (metrics['mean_portfolio'] - 100000) / 1000
    
    print(f"\n💰 PORTFOLIO:")
    print(f"   Moyen : ${metrics['mean_portfolio']:,.0f} ({profit_pct:+.1f}%)")
    print(f"   Min   : ${metrics['min_portfolio']:,.0f}")
    print(f"   Max   : ${metrics['max_portfolio']:,.0f}")
    
    print(f"\n📈 MÉTRIQUES:")
    print(f"   Sharpe: {metrics['sharpe']:.3f}")
    
    print(f"\n🎯 ACTIONS:")
    print(f"   HOLD  : {metrics['hold_pct']:5.1f}%")
    print(f"   BUY   : {metrics['buy_pct']:5.1f}%")
    print(f"   SELL  : {metrics['sell_pct']:5.1f}%")
    
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
    
    if metrics['sharpe'] >= 0.5:
        print("✅ Sharpe > 0.5 : PASS")
    else:
        print(f"❌ Sharpe : {metrics['sharpe']:.3f} < 0.5")
        success = False
    
    if metrics['buy_pct'] >= 10 and metrics['sell_pct'] >= 10:
        print("✅ Actions équilibrées : PASS")
    else:
        print(f"❌ Actions : BUY {metrics['buy_pct']:.1f}% / SELL {metrics['sell_pct']:.1f}%")
        success = False
    
    print("\n" + "="*80)
    
    if success:
        print("✅ TEST DISCRET : SUCCÈS !")
        print("\n🎉 L'IA a appris à trader avec actions discrètes !")
        print("   Signal clair: BUY/HOLD/SELL bien différenciés.")
        print("\n🚀 Prochaine étape:")
        print("   1. Appliquer ce pattern à UniversalTradingEnv")
        print("   2. Tester avec plusieurs assets")
        print("   3. Ajouter indicateurs (RSI, MACD)")
    else:
        print("❌ TEST DISCRET : ÉCHEC")
        print("\n🔧 Si ça échoue même avec actions discrètes:")
        print("   1. Vérifier que SELL donne vraiment un reward")
        print("   2. Augmenter ent_coef (exploration)")
        print("   3. Tester avec A2C au lieu de PPO")
        print("   4. Simplifier encore plus (1 seul BUY, 1 seul SELL par épisode)")
    
    print("="*80)
    
    return success

if __name__ == '__main__':
    print("\n🎯 Lance test actions discrètes...\n")
    
    success = main()
    
    exit(0 if success else 1)
