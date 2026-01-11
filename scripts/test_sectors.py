
import sys
import os
# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
import pandas as pd
from ploutos.env.environment import TradingEnvironment

def test_sectors():
    print("Testing Sector Diversification...")
    
    # 1. Create Dummy Data (10 Tickers)
    # We need correlation. Let's make Ticker 0-4 highly correlated, 5-9 highly correlated.
    N = 200
    dates = pd.date_range(start='2023-01-01', periods=N, freq='D')
    
    data = {}
    base_price_1 = 100 + np.cumsum(np.random.randn(N)) # Random Walk 1
    base_price_2 = 100 + np.cumsum(np.random.randn(N)) # Random Walk 2
    
    for i in range(10):
        if i < 5:
            price = base_price_1 + np.random.randn(N) * 0.1 # Cluster A
        else:
            price = base_price_2 + np.random.randn(N) * 0.1 # Cluster B
            
        df = pd.DataFrame(index=dates)
        df['Close'] = price
        df['High'] = price
        df['Low'] = price
        df['Open'] = price
        df['Volume'] = 1000
        data[f'TICKER_{i}'] = df
        
    # 2. Setup Env
    print("Initializing Environment...")
    env = TradingEnvironment(data=data)
    
    print(f"Clusters: {env.ticker_clusters}")
    print(f"Obs Space: {env.observation_space.shape}")
    
    # Validate Clusters
    # Ticker 0 and 1 should be same cluster
    if env.ticker_clusters[0] == env.ticker_clusters[1]:
        print("Cluster Logic seems consistent (0 and 1 match).")
    else:
        print("Cluster Logic divergence (0 and 1 differ).")
        
    obs, _ = env.reset(options={'current_step': 100})
    
    # 3. Buy Ticker 0
    print("Buying Ticker 0...")
    actions = [0] * 10
    actions[0] = 1 # Buy Ticker 0
    
    obs, _, _, _, _ = env.step(actions)
    
    # Check Sector Exposure
    cluster_idx = env.ticker_clusters[0]
    exposure = env.sector_exposure[cluster_idx]
    print(f"Sector {cluster_idx} Exposure: {exposure:.4f} (Should be > 0)")
    
    if exposure > 0.1:
         print("Sector Exposure Vector Updated Successfully!")
    else:
         print("Sector Exposure NOT found in obs.")

if __name__ == "__main__":
    test_sectors()
