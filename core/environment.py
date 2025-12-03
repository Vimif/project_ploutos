# core/environment.py
"""Environnement de trading pour RL"""

# === FIX PATH ===
import sys
from pathlib import Path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))
# ================

import gymnasium as gym
import numpy as np
import yfinance as yf
from core.features import FeatureCalculator
from config.settings import N_FEATURES

class TradingEnv(gym.Env):
    """Environnement de trading"""
    
    def __init__(self, ticker, df=None):
        super().__init__()
        
        self.ticker = ticker
        self.df = df
        
        if self.df is None:
            self.df = yf.download(ticker, period='2y', progress=False)
        
        if isinstance(self.df.columns, pd.MultiIndex):
            self.df.columns = self.df.columns.get_level_values(0)
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(N_FEATURES,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)
        
        self.current_step = 0
        self.max_steps = len(self.df) - 1
        self.position = 0
        self.entry_price = 0.0
        self.balance = 10000.0
        self.initial_balance = 10000.0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = np.random.randint(50, min(len(self.df) - 100, self.max_steps))
        self.position = 0
        self.entry_price = 0.0
        self.balance = self.initial_balance
        return self._get_observation(), {}
    
    def _get_observation(self):
        features = FeatureCalculator.calculate(self.ticker, n_features=N_FEATURES)
        if features is None:
            features = np.zeros(N_FEATURES, dtype=np.float32)
        return features
    
    def step(self, action):
        current_price = float(self.df['Close'].iloc[self.current_step])
        reward = 0.0
        
        if action == 1 and self.position == 0:  # BUY
            self.position = 1
            self.entry_price = current_price
            reward = -0.001
        
        elif action == 2 and self.position == 1:  # SELL
            pnl = (current_price - self.entry_price) / self.entry_price
            reward = pnl - 0.001
            self.balance *= (1 + pnl)
            self.position = 0
            self.entry_price = 0.0
        
        if self.position == 1:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            reward += unrealized_pnl * 0.01
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        
        return self._get_observation(), reward, terminated, False, {'balance': self.balance}
    
    def render(self):
        pass
    
    def close(self):
        pass
