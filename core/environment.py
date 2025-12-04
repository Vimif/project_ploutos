import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
from core.data_loader import load_market_data

class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, csv_path=None, initial_balance=10000, commission=0.001, lookback_window=50):
        super(TradingEnv, self).__init__()

        self.df = load_market_data(csv_path)
        self.initial_balance = initial_balance
        self.commission = commission
        self.lookback_window = lookback_window

        # Ajout indicateurs
        self.df = self._add_technical_indicators(self.df)
        self.df = self.df.dropna()

        # Observation Space : 12 indicateurs * window + 2 portfolio state
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12 * lookback_window + 2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

    def _add_technical_indicators(self, df):
        # RSI
        rsi = RSIIndicator(close=df['Close'], window=14)
        df['RSI'] = rsi.rsi()

        # MACD
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()

        # Bollinger
        bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()

        # EMA
        df['EMA_50'] = EMAIndicator(close=df['Close'], window=50).ema_indicator()
        df['EMA_200'] = EMAIndicator(close=df['Close'], window=200).ema_indicator()
        
        return df

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares = 0
        self.current_step = self.lookback_window
        self.total_value = self.initial_balance
        return self._next_observation(), {}

    def _next_observation(self):
        frame = self.df.iloc[self.current_step - self.lookback_window : self.current_step]
        
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MACD_signal', 
                   'BB_High', 'BB_Low', 'EMA_50', 'EMA_200']
        
        obs = frame[features].values.flatten()
        
        # Normalisation par le dernier prix de clôture
        current_close = frame['Close'].iloc[-1] + 1e-8
        obs = obs / current_close
        
        # Portfolio state
        portfolio_obs = np.array([
            self.balance / self.initial_balance,
            (self.shares * self.df.iloc[self.current_step]['Close']) / self.initial_balance
        ])
        
        return np.concatenate([obs, portfolio_obs]).astype(np.float32)

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        prev_value = self.balance + self.shares * current_price

        if action == 1: # BUY
            max_shares = int(self.balance / (current_price * (1 + self.commission)))
            if max_shares > 0:
                self.balance -= max_shares * current_price * (1 + self.commission)
                self.shares += max_shares
        
        elif action == 2: # SELL
            if self.shares > 0:
                self.balance += self.shares * current_price * (1 - self.commission)
                self.shares = 0
        
        self.current_step += 1
        new_price = self.df.iloc[self.current_step]['Close']
        current_value = self.balance + self.shares * new_price
        
        # Reward = Log Return pour stabilité
        reward = np.log(current_value / (prev_value + 1e-8))
        
        terminated = self.current_step >= len(self.df) - 1
        if current_value < self.initial_balance * 0.5: # Stop loss global -50%
            terminated = True
            reward = -1.0

        return self._next_observation(), reward, terminated, False, {'total_value': current_value}
