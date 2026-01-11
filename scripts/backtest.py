
import argparse
import os
import sys
import glob

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from ploutos.env.environment import TradingEnvironment

def run_backtest(model_path, data_file, tickers=None, period_start=None, period_end=None):
    print(f"Starting Backtest...")
    print(f"Model: {model_path}")
    print(f"Data: {data_file}")
    
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv(data_file, header=[0, 1], index_col=0, parse_dates=True)
    
    # Filter Tickers?
    if tickers:
        print(f"Filtering for tickers: {tickers}")
        # Reconstruct MultiIndex for specific tickers
        # Assuming level 1 is Ticker
        # This is a bit complex with MultiIndex, simpler to just pass all and let env handle it
        # but for visualization usually we want specific ones.
        # For now, let's load ALL, and env will process.
        pass
        
    # Filter Period?
    if period_start and period_end:
        print(f"Filtering period: {period_start} to {period_end}")
        df = df.loc[period_start:period_end]
    
    # Convert to Dict for Env
    data_dict = {}
    tickers_found = df.columns.get_level_values(1).unique()
    for ticker in tickers_found:
        data_dict[ticker] = df.xs(ticker, axis=1, level=1)
        
    # 2. Setup Env
    print("Creating Environment...")
    env = TradingEnvironment(data=data_dict)
    
    # 3. Load Model
    print("Loading Model...")
    model = PPO.load(model_path)
    
    # Get model's expected observation shape
    model_obs_shape = model.observation_space.shape[0]
    env_obs_shape = env.observation_space.shape[0]
    print(f"Model expects: {model_obs_shape} dims, Env produces: {env_obs_shape} dims")
    
    # Create adapter function if shapes mismatch
    if model_obs_shape != env_obs_shape:
        print(f"WARNING: Observation shape mismatch! Creating adapter...")
        def adapt_obs(obs):
            if len(obs) > model_obs_shape:
                return obs[:model_obs_shape]  # Truncate extra features
            elif len(obs) < model_obs_shape:
                return np.pad(obs, (0, model_obs_shape - len(obs)))  # Pad with zeros
            return obs
    else:
        adapt_obs = lambda x: x
    
    # 4. Run Simulation
    print("Running Simulation...")
    # Start from index 0 for backtest
    obs, info = env.reset(options={'current_step': 0})
    obs = adapt_obs(obs)  # Adapt initial observation
    done = False
    
    history = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        obs = adapt_obs(obs)  # Adapt each step's observation
        
        # Capture Step Data
        step_date = env.dates[env.current_step]
        
        # Portfolio Snapshots
        snapshot = {
            'date': step_date,
            'equity': env.equity,
            'balance': env.balance,
            'positions': env.portfolio_qty.copy(),
            'prices': env.prices_array[env.current_step].copy()
        }
        history.append(snapshot)
        
        if env.current_step % 100 == 0:
            print(f"Step {env.current_step}/{env.max_steps} | Equity: ${env.equity:.2f}")

    # 5. Process Results
    print("Processing Results...")
    df_res = pd.DataFrame(history)
    df_res.set_index('date', inplace=True)
    
    # 6. Visualization
    plot_results(df_res, env.tickers)

def plot_results(df, tickers):
    # Plot Equity Curve
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['equity'], label='Portfolio Equity', color='blue')
    plt.title('Backtest Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('backtest_equity.png')
    print("Saved backtest_equity.png")
    
    # Plot Trades for first 3 tickers
    for ticker in tickers[:3]:
        idx = tickers.index(ticker)
        
        # Extract price series
        prices = [step['prices'][idx] for step in df.to_dict('records')]
        pos = [step['positions'][idx] for step in df.to_dict('records')]
        dates = df.index
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, prices, label=f'{ticker} Price', color='black', alpha=0.5)
        
        # Identify Buy/Sell points
        # Change in position > 0 is Buy, < 0 is Sell
        pos_arr = np.array(pos)
        pos_diff = np.diff(pos_arr, prepend=0)
        
        buys = np.where(pos_diff > 0)[0]
        sells = np.where(pos_diff < 0)[0]
        
        if len(buys) > 0:
            plt.scatter(dates[buys], np.array(prices)[buys], marker='^', color='green', label='Buy', s=100)
        if len(sells) > 0:
            plt.scatter(dates[sells], np.array(prices)[sells], marker='v', color='red', label='Sell', s=100)
            
        plt.title(f'Trade Analysis: {ticker}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'backtest_trades_{ticker}.png')
        print(f"Saved backtest_trades_{ticker}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model zip")
    parser.add_argument("--data", type=str, default="data/sp500.csv", help="Path to data csv")
    parser.add_argument("--tickers", type=str, nargs="+", help="Specific tickers to filter")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    
    args = parser.parse_args()
    
    run_backtest(args.model, args.data, args.tickers, args.start, args.end)
