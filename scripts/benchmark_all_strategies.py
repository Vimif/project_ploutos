"""
Benchmark automatique - VERSION FINALE FONCTIONNELLE
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
N_ENVS = 16
TIMESTEPS = 1_000_000

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
# HELPER : Conversion action → int
# ========================================

def action_to_int(action):
    """Convertit n'importe quel type d'action en int"""
    if isinstance(action, np.ndarray):
        # array(2) → 2
        return int(action.item())
    else:
        return int(action)

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
    
    if config['algo'] == PPO:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device="cuda",
            learning_rate=1e-4,
            batch_size=2048,
            n_steps=1024,
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
            buffer_size=50_000,
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
        
        # FIX : Conversion universelle
        action_int = action_to_int(action)
        
        obs, reward, terminated, truncated, info = env.step(action_int)
        
        portfolio_values.append(info['total_value'])
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
    final_value = portfolio_values[-1] if len(portfolio_values) > 0 else initial_value
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
        'HOLD': action_counts.get(0, 0) / total_days * 100 if total_days > 0 else 0,
        'BUY': action_counts.get(1, 0) / total_days * 100 if total_days > 0 else 0,
        'SELL': action_counts.get(2, 0) / total_days * 100 if total_days > 0 else 0
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
    
    if len(all_results) == 0:
        print("\n⚠️ Aucun résultat à afficher")
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = "reports/benchmarks"
    os.makedirs(report_dir, exist_ok=True)
    
    md_path = f"{report_dir}/benchmark_{TICKER}_{timestamp}.md"
    
    with open(md_path, 'w') as f:
        f.write(f"# 📊 BENCHMARK REPORT - {TICKER}\n\n")
        f.write(f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Ticker** : {TICKER}\n")
        f.write(f"**Training Steps** : {TIMESTEPS:,}\n")
        f.write(f"**Test Period** : 90 jours\n\n")
        
        f.write("---\n\n")
        
        f.write("## 🏆 CLASSEMENT DES STRATÉGIES\n\n")
        f.write("| Rang | Stratégie | Return | Sharpe | Max DD | Win Rate |\n")
        f.write("|------|-----------|--------|--------|--------|----------|\n")
        
        sorted_results = sorted(all_results.items(), 
                               key=lambda x: x[1]['sharpe_ratio'], 
                               reverse=True)
        
        for rank, (name, res) in enumerate(sorted_results, 1):
            emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "📊"
            f.write(f"| {emoji} {rank} | **{name}** | "
                   f"{res['total_return_pct']:+.2f}% | "
                   f"{res['sharpe_ratio']:.2f} | "
                   f"{res['max_drawdown_pct']:.2f}% | "
                   f"{res['win_rate_pct']:.1f}% |\n")
        
        f.write(f"| 📈 | **Buy & Hold** | {buy_hold_result['total_return_pct']:+.2f}% | N/A | N/A | N/A |\n\n")
        
        f.write("---\n\n")
        f.write("## 📋 DÉTAILS\n\n")
        
        for name, res in sorted_results:
            f.write(f"### {name}\n\n")
            f.write(f"- Return : {res['total_return_pct']:+.2f}%\n")
            f.write(f"- Sharpe : {res['sharpe_ratio']:.2f}\n")
            f.write(f"- Sortino : {res['sortino_ratio']:.2f}\n")
            f.write(f"- Win Rate : {res['win_rate_pct']:.1f}%\n\n")
            f.write(f"**Actions** : HOLD {res['action_distribution']['HOLD']:.0f}%, "
                   f"BUY {res['action_distribution']['BUY']:.0f}%, "
                   f"SELL {res['action_distribution']['SELL']:.0f}%\n\n")
        
        f.write("---\n\n")
        best = sorted_results[0]
        f.write(f"🏆 **Meilleure** : {best[0]}\n\n")
        f.write(f"- Sharpe {best[1]['sharpe_ratio']:.2f}\n")
        f.write(f"- Return {best[1]['total_return_pct']:+.2f}%\n")
        f.write(f"- vs Buy&Hold : {best[1]['total_return_pct'] - buy_hold_result['total_return_pct']:+.2f}%\n")
    
    json_path = f"{report_dir}/benchmark_{TICKER}_{timestamp}.json"
    
    with open(json_path, 'w') as f:
        json.dump({
            'metadata': {'ticker': TICKER, 'timestamp': timestamp, 'steps': TIMESTEPS},
            'results': all_results,
            'buy_hold': buy_hold_result
        }, f, indent=2)
    
    print(f"\n✅ Rapport : {md_path}")
    return md_path

# ========================================
# MAIN
# ========================================

def main():
    print("\n🚀 BENCHMARK AUTOMATIQUE")
    print(f"📊 {len(STRATEGIES)} stratégies\n")
    
    all_results = {}
    
    for strategy_name, config in STRATEGIES.items():
        try:
            model_path = train_strategy(strategy_name, config)
            results = backtest_strategy(model_path, config['env_class'], config['algo'])
            all_results[config['name']] = results
            print(f"✅ {config['name']} : {results['total_return_pct']:+.2f}%, Sharpe {results['sharpe_ratio']:.2f}")
        except Exception as e:
            print(f"❌ {strategy_name} : {e}")
            import traceback
            traceback.print_exc()
    
    buy_hold = calculate_buy_and_hold(CSV_PATH)
    generate_report(all_results, buy_hold)
    print("\n🎉 Terminé !\n")

if __name__ == "__main__":
    main()