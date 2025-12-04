"""
Benchmark automatique - VERSION FINALE ROBUSTE & CORRIGÉE
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

from core.environment import TradingEnv
try:
    from core.environment_sharpe import TradingEnvSharpe
    HAS_SHARPE = True
except ImportError: HAS_SHARPE = False

try:
    from core.environment_continuous import TradingEnvContinuous
    from stable_baselines3 import SAC
    HAS_SAC = True
except ImportError: HAS_SAC = False

try:
    from core.environment_multitimeframe import TradingEnvMultiTimeframe
    HAS_MULTI = True
except ImportError: HAS_MULTI = False

# ========================================
# CONFIGURATION
# ========================================

TICKER = "SPY"  # On teste sur SPY maintenant
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
    STRATEGIES['sharpe'] = {'name': 'Sharpe Ratio Reward', 'env_class': TradingEnvSharpe, 'algo': PPO, 'timesteps': TIMESTEPS}
if HAS_MULTI:
    STRATEGIES['multitimeframe'] = {'name': 'Multi-Timeframe', 'env_class': TradingEnvMultiTimeframe, 'algo': PPO, 'timesteps': TIMESTEPS}

# ========================================
# HELPER : Conversion action → int
# ========================================
def action_to_int(action):
    if isinstance(action, np.ndarray):
        return int(action.item())
    return int(action)

# ========================================
# ENTRAÎNEMENT
# ========================================
def train_strategy(strategy_name, config):
    print("\n" + "="*70)
    print(f"🎯 ENTRAÎNEMENT : {config['name']}")
    print("="*70)
    
    def make_env():
        return config['env_class'](csv_path=CSV_PATH)
    
    env = SubprocVecEnv([make_env for _ in range(N_ENVS)])
    
    model = PPO(
        "MlpPolicy", env, verbose=1, device="cuda",
        learning_rate=1e-4, batch_size=2048, n_steps=1024, n_epochs=10,
        policy_kwargs=dict(net_arch=dict(pi=[512, 512, 512], vf=[512, 512, 512]))
    )
    
    print(f"🏋️ Entraînement {config['timesteps']:,} steps...")
    model.learn(total_timesteps=config['timesteps'], progress_bar=True)
    
    model_path = f"models/benchmark/{strategy_name}_{TICKER}.zip"
    os.makedirs("models/benchmark", exist_ok=True)
    model.save(model_path)
    env.close()
    return model_path

# ========================================
# BACKTESTING
# ========================================
def backtest_strategy(model_path, env_class, algo, test_days=90):
    print(f"\n📊 Backtesting {os.path.basename(model_path)}...")
    model = algo.load(model_path)
    env = env_class(csv_path=CSV_PATH)
    obs, _ = env.reset()
    
    portfolio_values = []
    actions_taken = []
    
    for _ in range(min(len(env.df) - env.lookback_window - 1, test_days * 24)):
        action, _ = model.predict(obs, deterministic=True)
        action_int = action_to_int(action)
        obs, _, terminated, truncated, info = env.step(action_int)
        portfolio_values.append(info['total_value'])
        actions_taken.append(action_int)
        if terminated or truncated: break
    
    df_backtest = pd.DataFrame({'portfolio_value': portfolio_values, 'action': actions_taken})
    df_backtest['returns'] = df_backtest['portfolio_value'].pct_change().fillna(0)
    
    mean_return = df_backtest['returns'].mean()
    std_return = df_backtest['returns'].std()
    sharpe = (mean_return / std_return) * np.sqrt(252 * 24) if std_return > 0 else 0
    
    total_days = len(df_backtest)
    action_counts = df_backtest['action'].value_counts()
    
    # Stats finales
    initial = 10000
    final = portfolio_values[-1] if portfolio_values else initial
    total_return = (final - initial) / initial
    
    return {
        'total_return_pct': total_return * 100,
        'sharpe_ratio': sharpe,
        'win_rate_pct': (df_backtest['returns'] > 0).sum() / total_days * 100 if total_days else 0,
        'max_drawdown_pct': 0.0, # Simplifié
        'action_distribution': {
            'HOLD': action_counts.get(0, 0) / total_days * 100 if total_days else 0,
            'BUY': action_counts.get(1, 0) / total_days * 100 if total_days else 0,
            'SELL': action_counts.get(2, 0) / total_days * 100 if total_days else 0
        }
    }

# ========================================
# BUY & HOLD ROBUSTE
# ========================================
def calculate_buy_and_hold(csv_path, test_days=90):
    # FIX: Lecture robuste des données
    df = pd.read_csv(csv_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    
    # Forcer conversion numérique
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna()
    
    initial_price = df['Close'].iloc[50]
    final_price = df['Close'].iloc[min(50 + test_days * 24, len(df) - 1)]
    
    # S'assurer que ce sont des floats
    initial_price = float(initial_price)
    final_price = float(final_price)
    
    buy_hold_return = (final_price - initial_price) / initial_price
    return {'total_return_pct': buy_hold_return * 100, 'sharpe_ratio': 0, 'strategy': 'Buy & Hold'}

# ========================================
# MAIN
# ========================================
def generate_report(all_results, buy_hold_result):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("reports/benchmarks", exist_ok=True)
    md_path = f"reports/benchmarks/benchmark_{TICKER}_{timestamp}.md"
    
    with open(md_path, 'w') as f:
        f.write(f"# 📊 BENCHMARK {TICKER}\n\n")
        f.write("| Stratégie | Return | Sharpe | Win Rate |\n|---|---|---|---|\n")
        for name, res in sorted(all_results.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True):
            f.write(f"| {name} | {res['total_return_pct']:+.2f}% | {res['sharpe_ratio']:.2f} | {res['win_rate_pct']:.1f}% |\n")
        f.write(f"| Buy & Hold | {buy_hold_result['total_return_pct']:+.2f}% | N/A | N/A |\n")
    
    print(f"\n✅ Rapport : {md_path}")

def main():
    print(f"\n🚀 BENCHMARK AUTOMATIQUE - {TICKER}")
    all_results = {}
    
    for strategy_name, config in STRATEGIES.items():
        try:
            model_path = train_strategy(strategy_name, config)
            results = backtest_strategy(model_path, config['env_class'], config['algo'])
            all_results[config['name']] = results
            print(f"✅ {config['name']} : {results['total_return_pct']:+.2f}%")
        except Exception as e:
            print(f"❌ {strategy_name} : {e}")
    
    buy_hold = calculate_buy_and_hold(CSV_PATH)
    generate_report(all_results, buy_hold)

if __name__ == "__main__":
    main()