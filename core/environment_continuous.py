import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
import os

class TradingEnvContinuous(gym.Env):
    """Environnement avec actions continues (allocation de capital progressive)"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, csv_path=None, ticker=None, initial_balance=10000, lookback_window=50):
        super(TradingEnvContinuous, self).__init__()
        
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
            raise ValueError("TradingEnvContinuous doit recevoir 'csv_path' ou 'ticker'")
        
        self.df = self.df.dropna()
        
        if len(self.df) < lookback_window + 10:
            raise ValueError(f"Pas assez de données: {len(self.df)} lignes")
        
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        
        # NOUVEAU : Action continue [-1, +1]
        # -1 = Vendre 100% des actions
        #  0 = Ne rien faire (HOLD)
        # +1 = Investir 100% du cash disponible
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        obs_size = 5 * lookback_window + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.shares = 0
        self.current_step = self.lookback_window
        
        return self._next_observation(), {}
    
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
        """Exécute une action continue"""
        
        current_price = self.df.iloc[self.current_step]['Close']
        prev_value = self.balance + self.shares * current_price
        
        # ============================================
        # INTERPRÉTATION DE L'ACTION CONTINUE
        # ============================================
        action_value = np.clip(float(action[0]), -1.0, 1.0)  # Sécurité
        
        if action_value > 0.01:  # ACHETER (seuil pour éviter micro-trades)
            # Investir un % du cash disponible
            cash_to_invest = self.balance * action_value
            shares_to_buy = int(cash_to_invest // current_price)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                self.shares += shares_to_buy
                self.balance -= cost
                
        elif action_value < -0.01:  # VENDRE
            # Vendre un % des actions détenues
            shares_to_sell = int(self.shares * abs(action_value))
            
            if shares_to_sell > 0:
                proceeds = shares_to_sell * current_price
                self.balance += proceeds
                self.shares -= shares_to_sell
        
        # Si action_value proche de 0 → HOLD (ne rien faire)
        
        # AVANCER DANS LE TEMPS
        self.current_step += 1
        new_price = self.df.iloc[self.current_step]['Close']
        current_value = self.balance + self.shares * new_price
        
        # REWARD SIMPLE (performance step-by-step)
        reward = (current_value - prev_value) / (prev_value + 1e-8)
        
        # Pénalité légère pour inaction totale
        if abs(action_value) < 0.01:
            reward -= 0.0001
        
        # Bonus exploration (encourage à utiliser toute la gamme d'actions)
        if abs(action_value) > 0.5:  # Actions fortes
            reward += 0.0002
        
        # FIN D'ÉPISODE
        terminated = False
        if self.current_step >= len(self.df) - 1:
            terminated = True
            final_pnl = (current_value - self.initial_balance) / self.initial_balance
            reward += final_pnl * 10
        
        if current_value <= 0:
            terminated = True
            reward = -10.0
        
        truncated = False
        info = {
            "total_value": float(current_value),
            "action_value": float(action_value)
        }
        
        return self._next_observation(), reward, terminated, truncated, info
    
    def render(self, mode='human'):
        if self.current_step < len(self.df):
            current_price = self.df.iloc[self.current_step]['Close']
            total_value = self.balance + self.shares * current_price
            pnl = ((total_value - self.initial_balance) / self.initial_balance) * 100
            allocation = (self.shares * current_price) / total_value * 100 if total_value > 0 else 0
            
            print(f"Step {self.current_step} | Value: ${total_value:.2f} | P&L: {pnl:+.2f}% | "
                  f"Stock Allocation: {allocation:.1f}%")
