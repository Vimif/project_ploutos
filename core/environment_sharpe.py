import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
import os
from collections import deque

class TradingEnvSharpe(gym.Env):
    """Environnement avec Sharpe Ratio Reward - Récompense la qualité des gains"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, csv_path=None, ticker=None, initial_balance=10000, lookback_window=50):
        super(TradingEnvSharpe, self).__init__()
        
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
            raise ValueError("TradingEnvSharpe doit recevoir 'csv_path' ou 'ticker'")
        
        self.df = self.df.dropna()
        
        if len(self.df) < lookback_window + 10:
            raise ValueError(f"Pas assez de données: {len(self.df)} lignes")
        
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        
        # NOUVEAU : Historique des returns pour calculer Sharpe
        self.returns_history = deque(maxlen=100)  # 100 derniers steps
        
        # Espaces
        obs_size = 5 * lookback_window + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.shares = 0
        self.current_step = self.lookback_window
        self.done = False
        self.returns_history.clear()
        
        # NOUVEAU : Tracking buy&hold pour benchmark
        initial_price = self.df.iloc[self.lookback_window]['Close']
        self.buy_hold_shares = self.initial_balance / initial_price
        self.initial_price = initial_price
        
        observation = self._next_observation()
        info = {}
        return observation, info
    
    def _next_observation(self):
        """Construit l'observation courante"""
        frame = self.df.iloc[self.current_step - self.lookback_window:self.current_step]
        
        # OHLCV normalisé
        obs = frame[['Open', 'High', 'Low', 'Close', 'Volume']].values.flatten()
        current_close = frame['Close'].iloc[-1] + 1e-8
        obs = obs / current_close
        
        # État du portfolio
        current_price = self.df.iloc[self.current_step]['Close']
        portfolio_state = np.array([
            self.balance / self.initial_balance,
            (self.shares * current_price) / self.initial_balance,
            current_price / 1000.0
        ])
        
        return np.concatenate([obs, portfolio_state]).astype(np.float32)
    
    def step(self, action):
        """Exécute une action avec Sharpe Ratio Reward"""
        
        current_price = self.df.iloc[self.current_step]['Close']
        prev_value = self.balance + self.shares * current_price
        
        # EXÉCUTION DE L'ACTION
        if action == 1:  # BUY
            shares_to_buy = int(self.balance // current_price)
            if shares_to_buy > 0:
                self.shares += shares_to_buy
                self.balance -= shares_to_buy * current_price
                
        elif action == 2:  # SELL
            if self.shares > 0:
                self.balance += self.shares * current_price
                self.shares = 0
        
        # AVANCER DANS LE TEMPS
        self.current_step += 1
        new_price = self.df.iloc[self.current_step]['Close']
        current_value = self.balance + self.shares * new_price
        
        # ============================================
        # CALCUL DU SHARPE RATIO REWARD
        # ============================================
        
        # 1. Return de ce step
        step_return = (current_value - prev_value) / (prev_value + 1e-8)
        self.returns_history.append(step_return)
        
        # 2. Benchmark buy&hold
        buy_hold_value = self.buy_hold_shares * new_price
        buy_hold_return = (buy_hold_value / self.initial_balance) - 1
        
        # 3. Calcul du Sharpe (si assez d'historique)
        if len(self.returns_history) >= 30:
            returns_array = np.array(self.returns_history)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array) + 1e-8  # Éviter division par 0
            sharpe = mean_return / std_return
            
            # 4. Alpha (surplus vs buy&hold)
            current_return = (current_value / self.initial_balance) - 1
            alpha = current_return - buy_hold_return
            
            # REWARD FINAL = Sharpe pondéré + Alpha amplifié
            reward = (sharpe * 10) + (alpha * 5)
            
        else:
            # Phase d'exploration initiale (pas encore assez de données)
            reward = step_return * 10  # Simple return amplifié
            sharpe = 0.0
            alpha = 0.0
        
        # Pénalité légère pour inaction excessive
        if action == 0:  # HOLD
            reward -= 0.0002
        
        # Bonus pour diversifier les actions (éviter spam)
        if action != 0:
            reward += 0.0001
        
        # GESTION FIN D'ÉPISODE
        terminated = False
        if self.current_step >= len(self.df) - 1:
            terminated = True
            
            # Bonus final massif basé sur Sharpe total de l'épisode
            if len(self.returns_history) > 0:
                final_returns = np.array(self.returns_history)
                final_sharpe = np.mean(final_returns) / (np.std(final_returns) + 1e-8)
                final_pnl = (current_value / self.initial_balance) - 1
                
                # Combiner Sharpe et performance brute
                reward += (final_sharpe * 100) + (final_pnl * 20)
        
        # Ruine = Game Over avec pénalité sévère
        if current_value <= 0:
            terminated = True
            reward = -50.0
        
        truncated = False
        obs = self._next_observation()
        info = {
            "total_value": float(current_value),
            "sharpe": float(sharpe) if len(self.returns_history) >= 30 else 0.0,
            "alpha_vs_buyhold": float(alpha) if len(self.returns_history) >= 30 else 0.0,
            "buy_hold_value": float(buy_hold_value)
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        if self.current_step < len(self.df):
            current_price = self.df.iloc[self.current_step]['Close']
            total_value = self.balance + self.shares * current_price
            buy_hold_value = self.buy_hold_shares * current_price
            
            pnl = ((total_value - self.initial_balance) / self.initial_balance) * 100
            bh_pnl = ((buy_hold_value - self.initial_balance) / self.initial_balance) * 100
            
            print(f"Step {self.current_step} | Portfolio: ${total_value:.2f} ({pnl:+.2f}%) | "
                  f"Buy&Hold: ${buy_hold_value:.2f} ({bh_pnl:+.2f}%) | "
                  f"Alpha: {pnl - bh_pnl:+.2f}%")
