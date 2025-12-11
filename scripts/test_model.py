#!/usr/bin/env python3
"""
Script de Test / Backtest pour Ploutos V6
=========================================

Ce script :
1. Charge le modèle entraîné (final_model.zip)
2. Charge les données historiques
3. Exécute le modèle sur toute la période
4. Affiche les métriques clés (Sharpe, Return, Max DD, Win Rate)
5. Compare avec une stratégie Buy & Hold (SPY)

Usage:
    python scripts/test_model.py --model models/v6_test_5m/final_model.zip
"""

import argparse
import pandas as pd
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import os
import sys

# Import des modules internes
sys.path.append(os.getcwd())
from core.universal_environment_v6_better_timing import UniversalTradingEnvV6BetterTiming

def load_data(data_path):
    print(f"📂 Chargement des données depuis {data_path}...")
    df = pd.read_csv(data_path)
    data = {}
    if 'Ticker' in df.columns:
        for ticker, group in df.groupby('Ticker'):
            data[str(ticker)] = group.copy().reset_index(drop=True)
    else:
        data['UNKNOWN'] = df.copy()
    return data

def calculate_metrics(portfolio_history, initial_balance):
    values = np.array(portfolio_history)
    returns = np.diff(values) / values[:-1]
    
    total_return = (values[-1] - initial_balance) / initial_balance
    annualized_return = total_return * (252 / len(values)) if len(values) > 0 else 0
    
    # Sharpe Ratio (Annualisé)
    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    # Max Drawdown
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    max_drawdown = np.max(drawdown)
    
    return {
        "Total Return": f"{total_return:.2%}",
        "Ann. Return": f"{annualized_return:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"-{max_drawdown:.2%}",
        "Final Equity": f"${values[-1]:,.2f}"
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/v6_test_5m/final_model.zip', help='Chemin du modèle')
    parser.add_argument('--data', type=str, default='data/historical_daily.csv', help='Données')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ Erreur: Modèle introuvable à {args.model}")
        return

    # 1. Chargement Données & Modèle
    data = load_data(args.data)
    print(f"🤖 Chargement du modèle {args.model}...")
    model = PPO.load(args.model)

    # 2. Création Environnement
    env = UniversalTradingEnvV6BetterTiming(
        data=data,
        initial_balance=100000,
        commission=0.001,
        max_steps=len(next(iter(data.values()))) - 10 # Utiliser toute la durée
    )
    
    # Wrap TimeLimit (comme à l'entraînement)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=10000)

    # 3. Exécution du Backtest
    print("\n🚀 Démarrage du Backtest...")
    obs, _ = env.reset(seed=42)
    done = False
    truncated = False
    
    equity_curve = [100000]
    
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True) # Deterministic = Pas de hasard, pure stratégie
        obs, reward, done, truncated, info = env.step(action)
        equity_curve.append(info['equity'])
        
        if info['current_step'] % 100 == 0:
            print(f"   Step {info['current_step']}: Equity=${info['equity']:,.0f} (Trades: {info['total_trades']})")

    # 4. Résultats
    print("\n" + "="*50)
    print("📊 RÉSULTATS DU TEST (5M STEPS MODEL)")
    print("="*50)
    
    metrics = calculate_metrics(equity_curve, 100000)
    for k, v in metrics.items():
        print(f"{k:15}: {v}")
    
    print("-" * 50)
    print(f"Total Trades   : {info['total_trades']}")
    print(f"Winning Trades : {info['winning_trades']}")
    print(f"Win Rate       : {info['winning_trades']/info['total_trades']:.1%}" if info['total_trades'] > 0 else "Win Rate: N/A")
    print("="*50)

    # 5. Comparaison SPY (Buy & Hold)
    if 'SPY' in data:
        spy_data = data['SPY']
        # Ajuster les index pour correspondre à la période testée
        if len(spy_data) > len(equity_curve):
             spy_data = spy_data.iloc[-len(equity_curve):]
        
        spy_start = spy_data.iloc[0]['Close']
        spy_end = spy_data.iloc[-1]['Close']
        spy_return = (spy_end - spy_start) / spy_start
        print(f"🆚 Benchmark SPY Return: {spy_return:.2%}")
        
        if float(metrics['Total Return'].strip('%')) > spy_return * 100:
            print("✅ LE BOT BAT LE MARCHÉ! 🏆")
        else:
            print("⚠️ Le bot sous-performe le marché.")

if __name__ == "__main__":
    main()
