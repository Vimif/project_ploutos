"""
Benchmark automatique - VERSION FINALE ROBUSTE
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
    from core.environment_multitimeframe import TradingEnvMultiTimeframe
    HAS_MULTI = True
except ImportError: HAS_MULTI = False

# CONFIGURATION
TICKER = "SPY"
CSV_PATH = f"data_cache/{TICKER}.csv"
N_ENVS = 16
TIMESTEPS = 1_000_000

STRATEGIES = {
    'baseline': {'name': 'Baseline', 'env_class': TradingEnv, 'algo': PPO, 'timesteps': TIMESTEPS}
}
if HAS_SHARPE:
    STRATEGIES['sharpe'] = {'name': 'Sharpe Reward', 'env_class': TradingEnvSharpe, 'algo': PPO, 'timesteps': TIMESTEPS}
if HAS_MULTI:
    STRATEGIES['multi'] = {'name': 'Multi-Timeframe', 'env_class': TradingEnvMultiTimeframe, 'algo': PPO, 'timesteps': TIMESTEPS}

def action_to_int(action):
    if isinstance(action, np.ndarray):
        return int(action.item())
    return int(action)

def train_strategy(strategy_name, config):
    print(f"\n🎯 ENTRAÎNEMENT : {config['name']}")
    
    def make_env():
        return config['env_class'](csv_path=CSV_PATH)
    
    env = SubprocVecEnv([make_env for _ in range(N_ENVS)])
    
    model = PPO("MlpPolicy", env, verbose=1, device="cuda", batch_size=2048, n_steps=1024)
    model.learn(total_timesteps=config['timesteps'], progress_bar=True)
    
    path = f"models/benchmark/{strategy_name}_{TICKER}.zip"
    os.makedirs("models/benchmark", exist_ok=True)
    model.save(path)
    env.close()
    return path

def backtest_strategy(model_path, env_class, algo, test_days=90):
    print(f"\n📊 Backtesting {os.path.basename(model_path)}...")
    model = algo.load(model_path)
    env = env_class(csv_path=CSV_PATH)
    obs, _ = env.reset()
    
    values = []
    actions = []
    
    for _ in range(min(len(env.df) - env.lookback_window - 1, test_days * 24)):
        action, _ = model.predict(obs, deterministic=True)
        act_int = action_to_int(action)
        obs, _, done, trunc, info = env.step(act_int)
        values.append(info['total_value'])
        actions.append(act_int)
        if done or trunc: break
    
    df = pd.DataFrame({'value': values, 'action': actions})
    df['ret'] = df['value'].pct_change().fillna(0)
    
    sharpe = (df['ret'].mean() / df['ret'].std()) * np.sqrt(252*24) if df['ret'].std() > 0 else 0
    initial = 10000
    final = values[-1] if values else initial
    
    return {
        'return': (final - initial) / initial * 100,
        'sharpe': sharpe,
        'win_rate': (df['ret'] > 0).mean() * 100
    }

def main():
    print(f"\n🚀 BENCHMARK {TICKER}")
    
    # Vérifier/Télécharger Data
    if not os.path.exists(CSV_PATH):
        import yfinance as yf
        yf.download(TICKER, period="730d", interval="1h", progress=False).to_csv(CSV_PATH)
    
    results = {}
    for name, conf in STRATEGIES.items():
        try:
            path = train_strategy(name, conf)
            res = backtest_strategy(path, conf['env_class'], conf['algo'])
            results[name] = res
            print(f"✅ {name}: {res['return']:.2f}% | Sharpe: {res['sharpe']:.2f}")
        except Exception as e:
            print(f"❌ {name}: {e}")
            
    print("\n🎉 TERMINÉ")

if __name__ == "__main__":
    main()
