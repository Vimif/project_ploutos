"""
Benchmark automatique - VERSION ULTRA-ROBUSTE
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from core.environment import TradingEnv
try:
    from core.environment_sharpe import TradingEnvSharpe
    HAS_SHARPE = True
except ImportError: HAS_SHARPE = False

TICKER = "SPY"
CSV_PATH = f"data_cache/{TICKER}.csv"
N_ENVS = 16
TIMESTEPS = 1_000_000

STRATEGIES = {
    'baseline': {'name': 'Baseline', 'env_class': TradingEnv, 'algo': PPO, 'timesteps': TIMESTEPS}
}
if HAS_SHARPE:
    STRATEGIES['sharpe'] = {'name': 'Sharpe', 'env_class': TradingEnvSharpe, 'algo': PPO, 'timesteps': TIMESTEPS}

def action_to_int(action):
    return int(action.item()) if isinstance(action, np.ndarray) else int(action)

def train_strategy(name, config):
    print(f"\n🎯 {config['name']}")
    env = SubprocVecEnv([lambda: config['env_class'](csv_path=CSV_PATH) for _ in range(N_ENVS)])
    model = PPO("MlpPolicy", env, verbose=1, device="cuda", batch_size=2048, n_steps=1024)
    model.learn(total_timesteps=config['timesteps'], progress_bar=True)
    path = f"models/benchmark/{name}_{TICKER}.zip"
    os.makedirs("models/benchmark", exist_ok=True)
    model.save(path)
    env.close()
    return path

def backtest(path, env_class, algo):
    print(f"📊 Backtest {os.path.basename(path)}...")
    model = algo.load(path)
    env = env_class(csv_path=CSV_PATH)
    obs, _ = env.reset()
    
    values, actions = [], []
    for _ in range(min(len(env.df) - env.lookback_window - 1, 2160)):
        act, _ = model.predict(obs, deterministic=True)
        obs, _, done, trunc, info = env.step(action_to_int(act))
        values.append(info['total_value'])
        actions.append(action_to_int(act))
        if done or trunc: break
    
    df = pd.DataFrame({'value': values, 'action': actions})
    df['ret'] = df['value'].pct_change().fillna(0)
    sharpe = (df['ret'].mean() / df['ret'].std()) * np.sqrt(252*24) if df['ret'].std() > 0 else 0
    
    return {
        'return': (values[-1] - 10000) / 100 if values else 0,
        'sharpe': sharpe,
        'win_rate': (df['ret'] > 0).mean() * 100
    }

def main():
    print(f"\n🚀 BENCHMARK {TICKER}\n")
    
    # Téléchargement propre
    if not os.path.exists(CSV_PATH):
        print(f"📥 Téléchargement {TICKER}...")
        import yfinance as yf
        data = yf.download(TICKER, period="730d", interval="1h", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.xs(TICKER, axis=1, level=1)
        data.to_csv(CSV_PATH)
        print(f"✅ Sauvegardé : {CSV_PATH}\n")
    
    results = {}
    for name, conf in STRATEGIES.items():
        try:
            path = train_strategy(name, conf)
            res = backtest(path, conf['env_class'], conf['algo'])
            results[name] = res
            print(f"✅ {name}: Return {res['return']:.2f}%, Sharpe {res['sharpe']:.2f}\n")
        except Exception as e:
            print(f"❌ {name}: {e}\n")
    
    print("🎉 TERMINÉ !")

if __name__ == "__main__":
    main()
