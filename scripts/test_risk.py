
import sys
import os
# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
import pandas as pd
from ploutos.env.environment import TradingEnvironment

def test_risk_management():
    print("Testing Risk Management (Stop Loss)...")
    
    # 1. Create Dummy Data (Long enough to pass max_steps check)
    # We need > 100 steps. Let's make 300.
    N = 300
    dates = pd.date_range(start='2023-01-01', periods=N, freq='D')
    prices = [100.0] * N
    
    # Drop at step 105 (Assume we create env, start at 100)
    # Step 100: 100 (Buy)
    # Step 101: 100
    # Step 102: 99
    # Step 103: 97 (SL Trigger)
    
    prices[102] = 99.0
    prices[103] = 97.0
    prices[104] = 95.0
    
    df = pd.DataFrame(index=dates)
    df['Close'] = prices
    df['High'] = prices
    df['Low'] = prices
    df['Open'] = prices
    df['Volume'] = 1000
    
    data = {'TEST': df}
    
    # 2. Setup Env
    env = TradingEnvironment(data=data)
    env.stop_loss_pct = 0.02
    
    # Start at 100
    obs, _ = env.reset(options={'current_step': 100})
    
    # 3. Step 100: BUY
    print(f"Step 100 Price: {env.prices_array[100,0]}")
    action = [1] 
    obs, reward, done, _, info = env.step(action)
    
    print(f"Portfolio after Buy: Qty={env.portfolio_qty[0]}, Entry={env.entry_prices[0]}")
    
    # 4. Step 101: Hold (100)
    print(f"Step 101 Low: {env.low_array[101,0]}")
    env.step([0])
    
    # 5. Step 102: Hold (99)
    print(f"Step 102 Low: {env.low_array[102,0]}")
    env.step([0])
    
    # 6. Step 103: Hold (97) -> SL Trigger
    print(f"Step 103 Low: {env.low_array[103,0]}")
    env.step([0])
    
    print(f"Portfolio Final: {env.portfolio_qty[0]}")
    
    if env.portfolio_qty[0] == 0:
        print("Stop Loss Triggered Successfully!")
        print(f"Balance: {env.balance} (Should be < Initial due to loss)")
    else:
        print("Stop Loss FAILED! Position still held.")

if __name__ == "__main__":
    test_risk_management()
