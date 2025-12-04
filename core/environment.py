import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
import os

class TradingEnv(gym.Env):
    """Environnement de trading optimisé pour Stable-Baselines3 (Gymnasium)"""
    
    metadata = {'render_modes': ['human']}
    
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
            raise ValueError(f"Pas assez de données: {len(self.df)} lignes")
        
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        
        # Espaces
        obs_size = 5 * lookback_window + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.shares = 0
        self.current_step = self.lookback_window
        self.done = False
        
        observation = self._next_observation()
        info = {}
        return observation, info
    
    def _next_observation(self):
        """Construit l'observation courante"""
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
        """Exécute une action avec système de reward amélioré"""
        
        # Prix actuel et précédent
        current_price = self.df.iloc[self.current_step]['Close']
        prev_price = self.df.iloc[self.current_step - 1]['Close']
        
        # Valeur du portfolio AVANT l'action
        prev_value = self.balance + self.shares * prev_price
        
        # Exécution de l'action
        if action == 1:  # BUY
            shares_to_buy = int(self.balance // current_price)
            if shares_to_buy > 0:
                self.shares += shares_to_buy
                self.balance -= shares_to_buy * current_price
                
        elif action == 2:  # SELL
            if self.shares > 0:
                self.balance += self.shares * current_price
                self.shares = 0
        
        # Avancer dans le temps
        self.current_step += 1
        
        # Valeur du portfolio APRÈS l'action
        new_price = self.df.iloc[self.current_step]['Close']
        current_value = self.balance + self.shares * new_price
        
        # REWARD AMÉLIORÉ (step-by-step au lieu de global)
        # Encourage les décisions qui augmentent la valeur immédiatement
        reward = (current_value - prev_value) / (prev_value + 1e-8)
        
        # Bonus d'exploration : encourage à ne pas spammer HOLD
        if action != 0:  # Si BUY ou SELL
            reward += 0.0005  # Petit bonus pour prendre des décisions
        else:  # Si HOLD
            reward -= 0.0001  # Petite pénalité pour éviter l'inaction
        
        # Vérifier fin d'épisode
        terminated = False
        if self.current_step >= len(self.df) - 1:
            terminated = True
            # Bonus/malus final selon performance globale
            final_pnl = (current_value - self.initial_balance) / self.initial_balance
            reward += final_pnl * 10  # Amplifier le signal final
        
        # Ruine = Game Over
        if current_value <= 0:
            terminated = True
            reward = -10.0  # Pénalité forte pour ruine
        
        truncated = False
        obs = self._next_observation()
        info = {
            "total_value": float(current_value),
            "pnl_pct": float((current_value - self.initial_balance) / self.initial_balance * 100)
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        if self.current_step < len(self.df):
            current_price = self.df.iloc[self.current_step]['Close']
            total_value = self.balance + self.shares * current_price
            pnl = ((total_value - self.initial_balance) / self.initial_balance) * 100
            print(f"Step {self.current_step} | Value: ${total_value:.2f} | P&L: {pnl:+.2f}%")
