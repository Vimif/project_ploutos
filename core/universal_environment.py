"""
Environnement de trading universel
Peut trader N'IMPORTE QUEL asset avec les mêmes features
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from core.data_loader import load_market_data

class UniversalTradingEnv(gym.Env):
    """Environnement générique pour trader n'importe quel asset"""
    
    def __init__(self, tickers, regime_detector, initial_balance=10000):
        super().__init__()
        
        self.tickers = tickers  # Liste d'assets (ex: ['NVDA', 'MSFT', 'AAPL'])
        self.regime_detector = regime_detector
        self.initial_balance = initial_balance
        
        # Charger données de tous les tickers
        self.data = {}
        for ticker in tickers:
            try:
                self.data[ticker] = load_market_data(f"data_cache/{ticker}.csv")
            except:
                print(f"⚠️ {ticker} non disponible")
        
        # Observation : Features techniques + Régime + Portfolio
        # Features: 12 par asset * N assets + 5 (régime) + 3 (portfolio)
        n_assets = len(self.data)
        obs_size = (12 * 50 * n_assets) + 5 + (n_assets * 2)
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        
        # Actions : Pour chaque asset (HOLD, BUY, SELL)
        # → 3^N possibilités (mais on simplifie avec MultiDiscrete)
        self.action_space = spaces.MultiDiscrete([3] * n_assets)
        
    def reset(self, seed=None):
        # Initialisation portfolio
        self.balance = self.initial_balance
        self.positions = {ticker: 0 for ticker in self.tickers}
        self.current_step = 50
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """Construit l'observation multi-assets"""
        
        obs_parts = []
        
        # 1. Features de chaque asset
        for ticker in self.tickers:
            if ticker in self.data:
                df = self.data[ticker]
                frame = df.iloc[self.current_step - 50:self.current_step]
                # Features techniques (simplified)
                features = frame[['Close']].values.flatten()
                obs_parts.append(features)
        
        # 2. Régime de marché (encoded)
        regime_info = self.regime_detector.detect()
        regime_encoded = self._encode_regime(regime_info['regime'])
        obs_parts.append(regime_encoded)
        
        # 3. État du portfolio
        for ticker in self.tickers:
            if ticker in self.data:
                price = self.data[ticker].iloc[self.current_step]['Close']
                obs_parts.extend([
                    self.positions[ticker] * price / self.initial_balance,
                    self.balance / self.initial_balance
                ])
        
        return np.concatenate(obs_parts).astype(np.float32)
    
    def _encode_regime(self, regime):
        """Encode le régime en vecteur one-hot"""
        regimes = ['BULL', 'BEAR', 'SIDEWAYS', 'HIGH_VOLATILITY', 'UNKNOWN']
        encoded = np.zeros(5)
        if regime in regimes:
            encoded[regimes.index(regime)] = 1
        return encoded
    
    def step(self, actions):
        """Exécute les actions sur tous les assets"""
        
        prev_value = self.balance
        for ticker in self.tickers:
            if ticker in self.data:
                prev_value += self.positions[ticker] * self.data[ticker].iloc[self.current_step]['Close']
        
        # Exécuter chaque action
        for idx, ticker in enumerate(self.tickers):
            if ticker not in self.data:
                continue
            
            action = actions[idx]
            price = self.data[ticker].iloc[self.current_step]['Close']
            
            if action == 1:  # BUY
                shares = int(self.balance / (price * len(self.tickers)))
                if shares > 0:
                    self.positions[ticker] += shares
                    self.balance -= shares * price
            
            elif action == 2:  # SELL
                if self.positions[ticker] > 0:
                    self.balance += self.positions[ticker] * price
                    self.positions[ticker] = 0
        
        # Avancer
        self.current_step += 1
        
        # Calculer reward
        current_value = self.balance
        for ticker in self.tickers:
            if ticker in self.data:
                current_value += self.positions[ticker] * self.data[ticker].iloc[self.current_step]['Close']
        
        reward = np.log(current_value / (prev_value + 1e-8))
        
        terminated = self.current_step >= min(len(df) for df in self.data.values()) - 1
        
        return self._get_obs(), reward, terminated, False, {'total_value': current_value}
