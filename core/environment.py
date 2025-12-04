import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
import os

class TradingEnv(gym.Env):
    """Environnement de trading compatible avec Stable-Baselines3"""
    
    metadata = {'render.modes': ['human']}
    
    # --- MODIFICATION ICI : Ajout de l'argument csv_path ---
    def __init__(self, csv_path=None, ticker=None, initial_balance=10000, lookback_window=50):
        super(TradingEnv, self).__init__()
        
        # CHARGEMENT DES DONNÉES (Nouveau système de cache)
        if csv_path and os.path.exists(csv_path):
            # 1. Lecture ultra-rapide depuis le CSV local
            try:
                self.df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            except Exception as e:
                raise ValueError(f"Erreur lecture CSV {csv_path}: {e}")
                
        elif ticker:
            # 2. Fallback (Ancienne méthode)
            print(f"⚠️ Warning: Téléchargement direct de {ticker} (LENT)")
            self.df = yf.download(ticker, period="730d", interval="1h", auto_adjust=True, progress=False)
            if isinstance(self.df.columns, pd.MultiIndex):
                self.df = self.df.xs(ticker, axis=1, level=1)
        else:
            raise ValueError("TradingEnv doit recevoir 'csv_path' ou 'ticker'")
        
        # Nettoyage de base
        self.df = self.df.dropna()
        
        if len(self.df) < lookback_window + 10:
            raise ValueError(f"Pas assez de données: {len(self.df)} lignes (min requis: {lookback_window+10})")
        
        # Paramètres
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        
        # Espaces d'observation et d'action
        # [Open, High, Low, Close, Volume] * window + [balance, shares, current_price]
        obs_size = 5 * lookback_window + 3
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Initialisation
        self.reset()
    
    def reset(self):
        """Réinitialise l'environnement"""
        self.balance = self.initial_balance
        self.shares = 0
        # On commence après la fenêtre d'historique
        self.current_step = self.lookback_window
        self.done = False
        
        return self._next_observation()
    
    def _next_observation(self):
        """Construit l'observation courante"""
        # Fenêtre glissante
        frame = self.df.iloc[self.current_step - self.lookback_window:self.current_step]
        
        # Normalisation simple par le dernier prix de clôture
        obs = frame[['Open', 'High', 'Low', 'Close', 'Volume']].values.flatten()
        current_close = frame['Close'].iloc[-1] + 1e-8
        obs = obs / current_close
        
        # État du portefeuille
        current_price = self.df.iloc[self.current_step]['Close']
        portfolio_state = np.array([
            self.balance / self.initial_balance,
            (self.shares * current_price) / self.initial_balance,
            current_price / 1000.0 # Scaling arbitraire pour le prix
        ])
        
        return np.concatenate([obs, portfolio_state]).astype(np.float32)
    
    def step(self, action):
        """Exécute une action"""
        current_price = self.df.iloc[self.current_step]['Close']
        
        # Exécution
        if action == 1:  # BUY
            # On achète avec tout le cash disponible (simplification)
            # Ou une fraction, ici on va dire qu'on essaie d'acheter 1 action si on peut
            shares_to_buy = int(self.balance // current_price)
            if shares_to_buy > 0:
                self.shares += shares_to_buy
                self.balance -= shares_to_buy * current_price
                
        elif action == 2:  # SELL
            if self.shares > 0:
                self.balance += self.shares * current_price
                self.shares = 0
        
        # Avancer
        self.current_step += 1
        
        # Vérifier fin
        if self.current_step >= len(self.df) - 1:
            self.done = True
            
        # Calcul Reward : Variation du P&L
        total_value = self.balance + self.shares * current_price
        reward = (total_value - self.initial_balance) / self.initial_balance
        
        # Faillite
        if total_value <= 0:
            self.done = True
            reward = -1.0
            
        return self._next_observation(), reward, self.done, {}
    
    def render(self, mode='human'):
        current_price = self.df.iloc[self.current_step]['Close']
        total_value = self.balance + self.shares * current_price
        print(f"Step: {self.current_step}, Value: {total_value:.2f}")
