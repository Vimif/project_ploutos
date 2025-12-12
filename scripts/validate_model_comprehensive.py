#!/usr/bin/env python3
"""
Validation Comprehensive du Modèle Ploutos
==========================================

Ce script :
1. Divise les données en plusieurs périodes (Walk-Forward)
2. Teste le modèle sur chaque période
3. Compare avec des stratégies de base (Buy & Hold, Momentum, Random)
4. Génère des statistiques complètes
5. Détecte si le modèle overfit ou s'il est vraiment profitable

Usage:
    python scripts/validate_model_comprehensive.py \
        --model models/v6_test_5m/final_model.zip \
        --data data/historical_daily.csv \
        --periods 5
"""

import argparse
import pandas as pd
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import sys
import os
from datetime import datetime

sys.path.append(os.getcwd())
from core.universal_environment_v6_better_timing import UniversalTradingEnvV6BetterTiming

def load_data(data_path):
    """Charger les données"""
    df = pd.read_csv(data_path)
    data = {}
    if 'Ticker' in df.columns:
        for ticker, group in df.groupby('Ticker'):
            data[str(ticker)] = group.copy().reset_index(drop=True)
    else:
        data['UNKNOWN'] = df.copy()
    return data

def calculate_metrics(equity_curve):
    """Calculer les métriques de performance"""
    values = np.array(equity_curve)
    returns = np.diff(values) / values[:-1]
    
    total_return = (values[-1] - values[0]) / values[0]
    annual_return = total_return * (252 / len(values)) if len(values) > 0 else 0
    
    # Sharpe Ratio
    sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
    
    # Max Drawdown
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    max_drawdown = np.max(drawdown)
    
    # Sortino Ratio
    downside_returns = returns[returns < 0]
    sortino = (np.mean(returns) / np.std(downside_returns)) * np.sqrt(252) if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0
    
    return {
        'Total Return': total_return,
        'Annual Return': annual_return,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': -max_drawdown,
        'Final Equity': values[-1],
        'Num Steps': len(values),
    }

def get_underlying_env(env):
    """Récupérer l'env interne si wrappé (TimeLimit, etc.)"""
    while hasattr(env, 'env'):
        env = env.env
    return env

def run_model_backtest(model, env, num_steps=1000):
    """Exécuter un backtest du modèle"""
    obs, _ = env.reset(seed=42)
    equity_curve = [100000]
    done = False
    truncated = False
    step_count = 0
    
    while not (done or truncated) and step_count < num_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        equity_curve.append(info['equity'])
        step_count += 1
    
    return equity_curve, info

def run_buyhold_backtest(env, num_steps=1000):
    """Stratégie naive: Buy & Hold au début"""
    obs, _ = env.reset(seed=42)
    equity_curve = [100000]
    done = False
    truncated = False
    step_count = 0
    bought = False
    
    # Récupérer l'env interne
    underlying_env = get_underlying_env(env)
    n_tickers = len(underlying_env.tickers)
    
    while not (done or truncated) and step_count < num_steps:
        # Au premier step, acheter tout
        if not bought:
            action = np.array([1] * n_tickers)  # BUY tous les tickers
            bought = True
        else:
            action = np.array([0] * n_tickers)  # HOLD
        
        obs, reward, done, truncated, info = env.step(action)
        equity_curve.append(info['equity'])
        step_count += 1
    
    return equity_curve, info

def run_momentum_backtest(env, num_steps=1000):
    """Stratégie momentum simple: acheter si prix monte, vendre si baisse"""
    obs, _ = env.reset(seed=42)
    equity_curve = [100000]
    done = False
    truncated = False
    step_count = 0
    prev_prices = {}
    
    # Récupérer l'env interne
    underlying_env = get_underlying_env(env)
    
    while not (done or truncated) and step_count < num_steps:
        action = []
        for ticker in underlying_env.tickers:
            price = underlying_env._get_current_price(ticker)
            
            if ticker not in prev_prices:
                action.append(0)  # HOLD first time
            else:
                if price > prev_prices[ticker] * 1.01:  # +1% momentum
                    action.append(1)  # BUY
                elif price < prev_prices[ticker] * 0.99:  # -1% momentum
                    action.append(2)  # SELL
                else:
                    action.append(0)  # HOLD
            
            prev_prices[ticker] = price
        
        obs, reward, done, truncated, info = env.step(np.array(action))
        equity_curve.append(info['equity'])
        step_count += 1
    
    return equity_curve, info

def split_data_walkforward(data, n_periods=5):
    """
    Diviser les données en périodes walk-forward
    Retourne: [(train_data, test_data), ...]
    """
    # Récupérer la longueur totale
    total_length = len(next(iter(data.values())))
    period_length = total_length // n_periods
    
    periods = []
    for i in range(n_periods):
        start_idx = i * period_length
        end_idx = (i + 1) * period_length if i < n_periods - 1 else total_length
        
        period_data = {}
        for ticker, df in data.items():
            period_data[ticker] = df.iloc[start_idx:end_idx].reset_index(drop=True)
        
        periods.append({
            'period': i + 1,
            'data': period_data,
            'length': len(period_data[next(iter(period_data.keys()))]),
            'start': start_idx,
            'end': end_idx,
        })
    
    return periods

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/v6_test_5m/final_model.zip')
    parser.add_argument('--data', type=str, default='data/historical_daily.csv')
    parser.add_argument('--periods', type=int, default=5)
    args = parser.parse_args()
    
    # Chargement
    print("\n" + "="*70)
    print("🔍 VALIDATION COMPREHENSIVE DU MODÈLE PLOUTOS")
    print("="*70)
    
    data = load_data(args.data)
    print(f"\n📊 Données chargées: {len(data)} tickers, {len(next(iter(data.values())))} steps par ticker")
    
    if not os.path.exists(args.model):
        print(f"❌ Modèle introuvable: {args.model}")
        return
    
    model = PPO.load(args.model)
    print(f"✅ Modèle chargé: {args.model}")
    
    # Division walk-forward
    periods = split_data_walkforward(data, n_periods=args.periods)
    print(f"\n📈 Données divisées en {len(periods)} périodes de validation")
    
    # Résultats
    results = {
        'model': [],
        'buyhold': [],
        'momentum': [],
    }
    
    # Test sur chaque période
    print("\n" + "-"*70)
    print("BACKTEST PAR PÉRIODE")
    print("-"*70)
    
    for period in periods:
        print(f"\n📊 Période {period['period']}/{len(periods)} ({period['length']} steps)")
        
        try:
            # Environnement
            env_model = UniversalTradingEnvV6BetterTiming(
                data=period['data'],
                initial_balance=100000,
                max_steps=period['length']
            )
            env_model = gym.wrappers.TimeLimit(env_model, max_episode_steps=period['length'])
            
            # Test modèle
            equity_model, _ = run_model_backtest(model, env_model, num_steps=period['length'])
            metrics_model = calculate_metrics(equity_model)
            results['model'].append(metrics_model)
            print(f"  Model     Return: {metrics_model['Total Return']:.2%} | Sharpe: {metrics_model['Sharpe Ratio']:.2f} | Max DD: {metrics_model['Max Drawdown']:.2%}")
        except Exception as e:
            print(f"  Model     ERROR: {e}")
            results['model'].append({'Total Return': 0.0, 'Sharpe Ratio': 0.0, 'Max Drawdown': 0.0})
        
        try:
            # Test buy & hold
            env_bh = UniversalTradingEnvV6BetterTiming(
                data=period['data'],
                initial_balance=100000,
                max_steps=period['length']
            )
            env_bh = gym.wrappers.TimeLimit(env_bh, max_episode_steps=period['length'])
            equity_bh, _ = run_buyhold_backtest(env_bh, num_steps=period['length'])
            metrics_bh = calculate_metrics(equity_bh)
            results['buyhold'].append(metrics_bh)
            print(f"  Buy&Hold  Return: {metrics_bh['Total Return']:.2%} | Sharpe: {metrics_bh['Sharpe Ratio']:.2f} | Max DD: {metrics_bh['Max Drawdown']:.2%}")
        except Exception as e:
            print(f"  Buy&Hold  ERROR: {e}")
            results['buyhold'].append({'Total Return': 0.0, 'Sharpe Ratio': 0.0, 'Max Drawdown': 0.0})
        
        try:
            # Test momentum
            env_mom = UniversalTradingEnvV6BetterTiming(
                data=period['data'],
                initial_balance=100000,
                max_steps=period['length']
            )
            env_mom = gym.wrappers.TimeLimit(env_mom, max_episode_steps=period['length'])
            equity_mom, _ = run_momentum_backtest(env_mom, num_steps=period['length'])
            metrics_mom = calculate_metrics(equity_mom)
            results['momentum'].append(metrics_mom)
            print(f"  Momentum  Return: {metrics_mom['Total Return']:.2%} | Sharpe: {metrics_mom['Sharpe Ratio']:.2f} | Max DD: {metrics_mom['Max Drawdown']:.2%}")
        except Exception as e:
            print(f"  Momentum  ERROR: {e}")
            results['momentum'].append({'Total Return': 0.0, 'Sharpe Ratio': 0.0, 'Max Drawdown': 0.0})
        
        # Comparaison
        if results['model'][-1]['Total Return'] > results['buyhold'][-1]['Total Return']:
            beat = results['model'][-1]['Total Return'] - results['buyhold'][-1]['Total Return']
            print(f"  ✅ Model bat Buy&Hold par {beat:.2%}")
        else:
            beat = results['buyhold'][-1]['Total Return'] - results['model'][-1]['Total Return']
            print(f"  ❌ Model sous-performe Buy&Hold par {beat:.2%}")
    
    # Résumé global
    print("\n" + "="*70)
    print("📋 RÉSUMÉ GLOBAL")
    print("="*70)
    
    for strategy, metrics_list in results.items():
        valid_returns = [m['Total Return'] for m in metrics_list if m['Total Return'] != 0.0 or m['Sharpe Ratio'] != 0.0]
        if valid_returns:
            avg_return = np.mean(valid_returns)
            avg_sharpe = np.mean([m['Sharpe Ratio'] for m in metrics_list])
            avg_dd = np.mean([m['Max Drawdown'] for m in metrics_list])
            consistency = np.std(valid_returns)
            
            print(f"\n{strategy.upper()}:")
            print(f"  Avg Return:     {avg_return:.2%}")
            print(f"  Avg Sharpe:     {avg_sharpe:.2f}")
            print(f"  Avg Max DD:     {avg_dd:.2%}")
            print(f"  Consistency:    {consistency:.2%} (écart-type)")
    
    # Verdict
    print("\n" + "="*70)
    model_returns = [m['Total Return'] for m in results['model'] if m['Total Return'] != 0.0 or m['Sharpe Ratio'] != 0.0]
    bh_returns = [m['Total Return'] for m in results['buyhold'] if m['Total Return'] != 0.0 or m['Sharpe Ratio'] != 0.0]
    
    if model_returns and bh_returns:
        model_avg = np.mean(model_returns)
        bh_avg = np.mean(bh_returns)
        
        if model_avg > bh_avg * 1.1:
            print("✅ VERDICT: Le modèle est prometteur (+10% vs Buy&Hold)")
            print("   → Valide pour entraînement long terme (50M steps)")
        elif model_avg > bh_avg:
            print("⚠️  VERDICT: Le modèle performe mais légèrement seulement")
            print("   → Besoin d'optimisations avant long terme")
        else:
            print("❌ VERDICT: Le modèle sous-performe")
            print("   → Nécessite correction avant production")
    print("="*70)

if __name__ == "__main__":
    main()
