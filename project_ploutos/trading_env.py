# trading_env.py (VERSION V26 - AGRESSIVE & VISION LONGUE)
# ---------------------------------------------------------
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=10000):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        
        # Actions : 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # VISION AMÉLIORÉE : On regarde 5 jours en arrière
        # 6 Indicateurs x 5 Jours = 30 Neurones d'entrée
        self.window_size = 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(6 * self.window_size,), 
            dtype=np.float32
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.current_step = self.window_size # On commence après la fenêtre
        self.prev_net_worth = self.initial_balance
        return self._next_observation(), {}

    def _next_observation(self):
        # On construit un vecteur géant des 5 derniers jours
        # Pour que l'IA comprenne la dynamique (Est-ce que ça monte ou ça descend ?)
        end = self.current_step
        start = end - self.window_size
        
        obs_list = []
        
        for i in range(start, end):
            if i < 0:
                d = [0]*6 # Padding si début
            else:
                row = self.df.iloc[i]
                d = [
                    row['Close'] / 1000,       # Prix Normalisé
                    row.get('RSI', 50) / 100,  # RSI
                    row.get('SMA_Ratio', 1),   # Écart à la Moyenne
                    row.get('MACD', 0),        # MACD
                    self.balance / 10000,      # Mon Cash
                    1 if self.shares_held > 0 else 0 # Suis-je investi ?
                ]
            obs_list.extend(d)
            
        return np.array(obs_list, dtype=np.float32)

    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        # Calcul Valeur
        current_price = self.df.iloc[self.current_step]['Close']
        self.net_worth = self.balance + (self.shares_held * current_price)
        
        # --- REWARD ENGINEERING (PARTIE CRITIQUE) ---
        profit = self.net_worth - self.prev_net_worth
        reward = profit
        
        # 1. Bonus d'Investissement : On encourage à être DANS le marché
        if self.shares_held > 0:
            reward += 0.2
        
        # 2. Punition "FOMO" : Si on est Cash alors que ça monte
        else:
            prev_price = self.df.iloc[self.current_step-1]['Close']
            if current_price > prev_price:
                reward -= 1.0 # Grosse punition : "Tu rates le train !"

        # 3. Punition Stop-Loss : Si on perd trop d'un coup, on punit
        if profit < -500: # Grosse perte
            reward -= 10

        self.prev_net_worth = self.net_worth
        
        return self._next_observation(), reward, terminated, truncated, {}

    def _take_action(self, action):
        current_price = self.df.iloc[self.current_step]['Close']
        
        # BUY (1)
        if action == 1 and self.balance > current_price:
            # All-in pour maximiser l'impact (Simulation Agressive)
            shares = self.balance // current_price
            self.balance -= shares * current_price
            self.shares_held += shares
            
        # SELL (2)
        elif action == 2 and self.shares_held > 0:
            self.balance += self.shares_held * current_price
            self.shares_held = 0
