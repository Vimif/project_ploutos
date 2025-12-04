import gymnasium as gym  # <-- Changement ici
from gymnasium import spaces # <-- Changement ici
import numpy as np
import pandas as pd
import yfinance as yf
import os

class TradingEnv(gym.Env):
    """Environnement de trading compatible avec Stable-Baselines3 (Gymnasium)"""
    
    metadata = {'render_modes': ['human']} # <-- Changement de clé
    
    def __init__(self, csv_path=None, ticker=None, initial_balance=10000, lookback_window=50):
        super(TradingEnv, self).__init__()
        
        # CHARGEMENT DES DONNÉES
        if csv_path and os.path.exists(csv_path):
            try:
                self.df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            except Exception as e:
                raise ValueError(f"Erreur lecture CSV {csv_path}: {e}")
        elif ticker:
            print(f"⚠️ Warning: Téléchargement direct de {ticker} (LENT)")
            self.df = yf.download(ticker, period="730d", interval="1h", auto_adjust=True, progress=False)
            if isinstance(self.df.columns, pd.MultiIndex):
                self.df = self.df.xs(ticker, axis=1, level=1)
        else:
            raise ValueError("TradingEnv doit recevoir 'csv_path' ou 'ticker'")
        
        self.df = self.df.dropna()
        
        if len(self.df) < lookback_window + 10:
            raise ValueError(f"Pas assez de données: {len(self.df)} lignes (min requis: {lookback_window+10})")
        
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        
        # Espaces d'observation et d'action
        obs_size = 5 * lookback_window + 3
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(3)
        
    def reset(self, seed=None, options=None): # <-- Changement signature Gymnasium
        super().reset(seed=seed) # <-- Important pour le seeding
        
        self.balance = self.initial_balance
        self.shares = 0
        self.current_step = self.lookback_window
        self.done = False
        
        observation = self._next_observation()
        info = {} # Gymnasium demande (obs, info)
        return observation, info
    
    def _next_observation(self):
        frame = self.df.iloc[self.current_step - self.lookback_window:self.current_step]
        
        obs = frame[['Open', 'High', 'Low', 'Close', 'Volume']].values.flatten()
        current_close = frame['Close'].iloc[-1] + 1e-8
        obs = obs / current_close
        
        current_price = self.df.iloc[self.current_step]['Close']
        portfolio_state = np.array([
            self.balance / self.initial_balance,
            (self.shares * current_price) / self.initial_balance,
            current_price / 1000.0
        ])
        
        return np.concatenate([obs, portfolio_state]).astype(np.float32)
    
    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        prev_value = self.balance + self.shares * self.df.iloc[self.current_step - 1]['Close']
        
        # Actions...
        if action == 1:  # BUY
            shares_to_buy = int(self.balance // current_price)
            if shares_to_buy > 0:
                self.shares += shares_to_buy
                self.balance -= shares_to_buy * current_price
        elif action == 2:  # SELL
            if self.shares > 0:
                self.balance += self.shares * current_price
                self.shares = 0
        
        self.current_step += 1
        current_value = self.balance + self.shares * current_price
        
        # NOUVELLE REWARD : Variation step-by-step (encourage les mouvements intelligents)
        reward = (current_value - prev_value) / prev_value
        
        # Bonus pour diversifier (éviter HOLD spam)
        if action != 0:  # Si BUY ou SELL
            reward += 0.001  # Petit bonus d'exploration

        # Pénaliser HOLD prolongé
        if action == 0:
            reward -= 0.0001  # Petite pénalité pour encourager l'action
            
        # Gymnasium distingue "terminated" (fin naturelle/échec) et "truncated" (timelimit)
        truncated = False 
            
        obs = self._next_observation()
        info = {"total_value": total_value}
        
        return obs, reward, terminated, truncated, info # <-- Retourne 5 valeurs maintenant !
    
    def render(self, mode='human'):
        # ... code render inchangé ...
        pass
