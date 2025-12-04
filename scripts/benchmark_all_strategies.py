"""
Benchmark automatique - VERSION CORRIGÉE
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import json
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# Imports conditionnels
from core.environment import TradingEnv

try:
    from core.environment_sharpe import TradingEnvSharpe
    HAS_SHARPE = True
except ImportError:
    HAS_SHARPE = False

try:
    from core.environment_continuous import TradingEnvContinuous
    from stable_baselines3 import SAC
    HAS_SAC = True
except ImportError:
    HAS_SAC = False

try:
    from core.environment_multitimeframe import TradingEnvMultiTimeframe
    HAS_MULTI = True
except ImportError:
    HAS_MULTI = False

# ========================================
# CONFIGURATION
# ========================================

TICKER = "NVDA"
CSV_PATH = f"data_cache/{TICKER}.csv"
N_ENVS = 32
TIMESTEPS = 1_000_000

# Construction dynamique
STRATEGIES = {
    'baseline': {
        'name': 'Baseline (Indicateurs Techniques)',
        'env_class': TradingEnv,
        'algo': PPO,
        'timesteps': TIMESTEPS
    }
}

if HAS_SHARPE:
    STRATEGIES['sharpe'] = {
        'name': 'Sharpe Ratio Reward',
        'env_class': TradingEnvSharpe,
        'algo': PPO,
        'timesteps': TIMESTEPS
    }

if HAS_SAC:
    STRATEGIES['continuous'] = {
        'name': 'Actions Continues (SAC)',
        'env_class': TradingEnvContinuous,
        'algo': SAC,
        'timesteps': TIMESTEPS
    }

if HAS_MULTI:
    STRATEGIES['multitimeframe'] = {
        'name': 'Multi-Timeframe',
        'env_class': TradingEnvMultiTimeframe,
        'algo': PPO,
        'timesteps': TIMESTEPS
    }

# ========================================
# ENTRAÎNEMENT
# ========================================

def train_strategy(strategy_name, config):
    """Entraîne une stratégie"""
    
    print("\n" + "="*70)
    print(f"🎯 ENTRAÎNEMENT : {config['name']}")
    print("="*70)
    
    def make_env():
        return config['env_class'](csv_path=CSV_PATH)
    
    env = SubprocVecEnv([make_env for _ in range(N_ENVS)])
    
    policy_kwargs = dict(net_arch=dict(pi=[512, 512, 512], vf=[512, 512, 512]))
    
    # Paramètres adaptés selon l'algo
    if config['algo'] == PPO:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device="cuda",
            learning_rate=1e-4,
            batch_size=4096,
            n_steps=2048,
            n_epochs=10,
            policy_kwargs=policy_kwargs
        )
    else:  # SAC
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            device="cuda",
            learning_rate=3e-4,
            buffer_size=100_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            policy_kwargs=policy_kwargs
        )
    
    print(f"🏋️ Entraînement {config['timesteps']:,} steps...")
    model.learn(total_timesteps=config['timesteps'], progress_bar=True)
    
    model_path = f"models/benchmark/{strategy_name}_{TICKER}.zip"
    os.makedirs("models/benchmark", exist_ok=True)
    model.save(model_path)
    
    env.close()
    print(f"✅ Modèle sauvegardé : {model_path}")
    
    return model_path

# ========================================
# BACKTESTING
# ========================================

def backtest_strategy(model_path, env_class, algo, test_days=90):
    """Backtest détaillé"""
    
    print(f"\n📊 Backtesting {os.path.basename(model_path)}...")
    
    model = algo.load(model_path)
    env = env_class(csv_path=CSV_PATH)
    obs, _ = env.reset()
    
    portfolio_values = []
    actions_taken = []
    rewards = []
    
    total_steps = min(len(env.df) - env.lookback_window - 1, test_days * 24)
    
    for step in range(total_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        portfolio_values.append(info['total_value'])
        
        # FIX : Gérer actions continues
        if hasattr(action, '__iter__') and not isinstance(action, str):
            action_int = int(action[0]) if len(action) > 0 else 0
        else:
            action_int = int(action)
        
        actions_taken.append(action_int)
        rewards.append(reward)
        
        if terminated or truncated:
            break
    
    # Calcul métriques
    df_backtest = pd.DataFrame({
        'portfolio_value': portfolio_values,
        'action': actions_taken,
        'reward': rewards
    })
    
    initial_value = 10000
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_value) / initial_value
    
    df_backtest['returns'] = df_backtest['portfolio_value'].pct_change().fillna(0)
    
    mean_return = df_backtest['returns'].mean()
    std_return = df_backtest['returns'].std()
    sharpe = (mean_return / std_return) * np.sqrt(252 * 24) if std_return > 0 else 0
    
    cumulative = (1 + df_backtest['returns']).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    winning_days = (df_backtest['returns'] > 0).sum()
    total_days = len(df_backtest)
    win_rate = winning_days / total_days if total_days > 0 else 0
    
    action_counts = df_backtest['action'].value_counts()
    action_dist = {
        'HOLD': action_counts.get(0, 0) / total_days * 100,
        'BUY': action_counts.get(1, 0) / total_days * 100,
        'SELL': action_counts.get(2, 0) / total_days * 100
    }
    
    volatility = std_return * np.sqrt(252 * 24)
    calmar = abs(total_return / max_drawdown) if max_drawdown < 0 else 0
    
    downside_returns = df_backtest['returns'][df_backtest['returns'] < 0]
    downside_std = downside_returns.std()
    sortino = (mean_return / downside_std) * np.sqrt(252 * 24) if downside_std > 0 else 0
    
    results = {
        'total_return_pct': total_return * 100,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_drawdown * 100,
        'win_rate_pct': win_rate * 100,
        'volatility_pct': volatility * 100,
        'calmar_ratio': calmar,
        'sortino_ratio': sortino,
        'final_value': final_value,
        'total_trades': total_days,
        'action_distribution': action_dist,
        'avg_reward': df_backtest['reward'].mean()
    }
    
    return results

# ========================================
# BUY & HOLD
# ========================================

def calculate_buy_and_hold(csv_path, test_days=90):
    """Calcule performance buy & hold"""
    
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df = df.dropna()
    
    initial_price = df['Close'].iloc[50]
    final_price = df['Close'].iloc[min(50 + test_days * 24, len(df) - 1)]
    
    buy_hold_return = (final_price - initial_price) / initial_price
    
    return {
        'total_return_pct': buy_hold_return * 100,
        'sharpe_ratio': 0,
        'max_drawdown_pct': 0,
        'strategy': 'Buy & Hold'
    }

# ========================================
# RAPPORT
# ========================================

def generate_report(all_results, buy_hold_result):
    """Génère rapport Markdown + JSON"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = "reports/benchmarks"
    os.makedirs(report_dir, exist_ok=True)
    
    md_path = f"{report_dir}/benchmark_{TICKER}_{timestamp}.md"
    
    with open(md_path, 'w') as f:
        f.write(f"# 📊 BENCHMARK REPORT - {TICKER}\n\n")
        f.write(f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Ticker** : {TICKER}\n")
        f.write(f"**Training Steps** : {TIMESTEPS:,}\n")
        f.write(f"**Test Period** : 90 jours (2160 heures)\n\n")
        
        f.write("---\n\n")
        
        f.write("## 🏆 CLASSEMENT DES STRATÉGIES\n\n")
        f.write("| Rang | Stratégie | Return | Sharpe | Max DD | Win Rate | Volatilité |\n")
        f.write("|------|-----------|--------|--------|--------|----------|------------|\n")
        
        sorted_results = sorted(all_results.items(), 
                               key=lambda x: x[1]['sharpe_ratio'], 
                               reverse=True)
        
        for rank, (name, res) in enumerate(sorted_results, 1):
            emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "📊"
            f.write(f"| {emoji} {rank} | **{name}** | "
                   f"{res['total_return_pct']:+.2f}% | "
                   f"{res['sharpe_ratio']:.2f} | "
                   f"{res['max_drawdown_pct']:.2f}% | "
                   f"{res['win_rate_pct']:.1f}% | "
                   f"{res['volatility_pct']:.1f}% |\n")
        
        f.write(f"| 📈 | **Buy & Hold** | {buy_hold_result['total_return_pct']:+.2f}% | "
               f"N/A | N/A | N/A | N/A |\n\n")
        
        f.write("---\n\n")
        
        f.write("## 📋 DÉTAILS PAR STRATÉGIE\n\n")
        
        for name, res in sorted_results:
            f.write(f"### {name}\n\n")
            f.write(f"- **Return Total** : {res['total_return_pct']:+.2f}%\n")
            f.write(f"- **Sharpe Ratio** : {res['sharpe_ratio']:.2f}\n")
            f.write(f"- **Sortino Ratio** : {res['sortino_ratio']:.2f}\n")
            f.write(f"- **Calmar Ratio** : {res['calmar_ratio']:.2f}\n")
            f.write(f"- **Max Drawdown** : {res['max_drawdown_pct']:.2f}%\n")
            f.write(f"- **Win Rate** : {res['win_rate_pct']:.1f}%\n")
            f.write(f"- **Volatilité** : {res['volatility_pct']:.1f}%\n")
            f.write(f"- **Valeur Finale** : ${res['final_value']:,.2f}\n")
            f.write(f"- **Reward Moyen** : {res['avg_reward']:.4f}\n\n")
            
            f.write(f"**Distribution Actions** :\n")
            for action, pct in res['action_distribution'].items():
                f.write(f"- {action} : {pct:.1f}%\n")
            f.write("\n")
        
        f.write("---\n\n")
        f.write("## 🎯 RECOMMANDATION\n\n")
        
        best = sorted_results[0]
        f.write(f"**Meilleure stratégie** : **{best[0]}**\n\n")
        f.write(f"- Surperforme le Buy & Hold de **{best[1]['total_return_pct'] - buy_hold_result['total_return_pct']:+.2f}%**\n")
        f.write(f"- Sharpe Ratio exceptionnel de **{best[1]['sharpe_ratio']:.2f}**\n")
        f.write(f"- Win Rate de **{best[1]['win_rate_pct']:.1f}%**\n")
    
    json_path = f"{report_dir}/benchmark_{TICKER}_{timestamp}.json"
    
    export_data = {
        'metadata': {
            'ticker': TICKER,
            'timestamp': timestamp,
            'training_steps': TIMESTEPS,
            'test_days': 90
        },
        'results': all_results,
        'buy_hold': buy_hold_result
    }
    
    with open(json_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\n✅ Rapport généré :")
    print(f"   📄 Markdown : {md_path}")
    print(f"   📊 JSON : {json_path}")
    
    return md_path

# ========================================
# MAIN
# ========================================

def main():
    print("\n" + "="*70)
    print("🚀 BENCHMARK AUTOMATIQUE")
    print("="*70)
    print(f"📊 Ticker : {TICKER}")
    print(f"🏋️ Training : {TIMESTEPS:,} steps")
    print(f"📈 Backtest : 90 jours")
    print(f"🎯 Stratégies : {len(STRATEGIES)}")
    print("="*70)
    
    for name in STRATEGIES.keys():
        print(f"  ✅ {STRATEGIES[name]['name']}")
    
    print("="*70)
    
    all_results = {}
    
    for strategy_name, config in STRATEGIES.items():
        try:
            model_path = train_strategy(strategy_name, config)
            results = backtest_strategy(model_path, config['env_class'], config['algo'])
            all_results[config['name']] = results
            
            print(f"\n✅ {config['name']} : Return {results['total_return_pct']:+.2f}%, Sharpe {results['sharpe_ratio']:.2f}")
            
        except Exception as e:
            print(f"\n❌ Erreur {strategy_name} : {e}")
            import traceback
            traceback.print_exc()
            continue
    
    buy_hold_result = calculate_buy_and_hold(CSV_PATH)
    print(f"\n📈 Buy & Hold : Return {buy_hold_result['total_return_pct']:+.2f}%")
    
    report_path = generate_report(all_results, buy_hold_result)
    
    print("\n" + "="*70)
    print("🎉 BENCHMARK TERMINÉ !")
    print("="*70)
    print(f"\n📄 Consulter : {report_path}\n")

if __name__ == "__main__":
    main()
