"""
Environnement de trading universel pour multi-assets
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class UniversalTradingEnv(gym.Env):
    """Environnement générique pour trader un portfolio multi-assets"""

    metadata = {'render_modes': ['human']}

    def __init__(self, data, initial_balance=100000, commission=0.001, max_steps=1000):
        """
        Args:
            data (dict): {ticker: DataFrame} avec colonnes [Open, High, Low, Close, Volume]
            initial_balance (float): Capital initial
            commission (float): Commission par trade (0.001 = 0.1%)
            max_steps (int): Nombre max de steps par épisode
        """
        super().__init__()
        
        self.data = data
        self.tickers = list(data.keys())
        self.n_assets = len(self.tickers)
        self.initial_balance = initial_balance
        self.commission = commission
        self.max_steps = max_steps
        
        # Vérifier que tous les DataFrames ont la même longueur
        self.data_length = min(len(df) for df in data.values())
        
        # Observation space : features par asset + portfolio state
        # Features par asset : [close_norm, volume_norm, rsi, macd, returns_1d, returns_5d]
        n_features_per_asset = 6
        n_portfolio_features = 3  # [cash_ratio, total_value_norm, n_positions]
        
        obs_size = self.n_assets * n_features_per_asset + n_portfolio_features
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # Action space : position pour chaque asset [-1, 1]
        # -1 = short max, 0 = neutral, 1 = long max
        self.action_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(self.n_assets,), 
            dtype=np.float32
        )
        
        # État interne
        self.current_step = 0
        self.balance = initial_balance
        self.positions = {ticker: 0 for ticker in self.tickers}  # Nombre d'actions
        self.portfolio_value = initial_balance
        self.trades_history = []
        
    def reset(self, seed=None, options=None):
        """Reset environnement"""
        super().reset(seed=seed)
        
        # Choisir point de départ aléatoire (laisser 100 steps pour calcul features)
        self.current_step = np.random.randint(100, self.data_length - self.max_steps)
        
        self.balance = self.initial_balance
        self.positions = {ticker: 0 for ticker in self.tickers}
        self.portfolio_value = self.initial_balance
        self.trades_history = []
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Exécuter une action"""
        self.current_step += 1
        
        # Récupérer prix actuels
        current_prices = {
            ticker: float(self.data[ticker].iloc[self.current_step]['Close'])
            for ticker in self.tickers
        }
        
        # Calculer valeur actuelle du portfolio
        positions_value = sum(
            self.positions[ticker] * current_prices[ticker]
            for ticker in self.tickers
        )
        self.portfolio_value = self.balance + positions_value
        
        # Exécuter actions (ajuster positions selon action)
        for i, ticker in enumerate(self.tickers):
            target_position = action[i]  # -1 à 1
            
            # Convertir en valeur monétaire cible
            target_value = target_position * self.portfolio_value * 0.95 / self.n_assets
            
            current_value = self.positions[ticker] * current_prices[ticker]
            trade_value = target_value - current_value
            
            # Exécuter trade si significatif
            if abs(trade_value) > self.portfolio_value * 0.01:  # Min 1% portfolio
                shares_to_trade = int(trade_value / current_prices[ticker])
                
                if shares_to_trade != 0:
                    cost = abs(shares_to_trade * current_prices[ticker])
                    commission_cost = cost * self.commission
                    
                    # Vérifier si assez de cash
                    if shares_to_trade > 0:  # Achat
                        if self.balance >= cost + commission_cost:
                            self.positions[ticker] += shares_to_trade
                            self.balance -= (cost + commission_cost)
                            self.trades_history.append({
                                'step': self.current_step,
                                'ticker': ticker,
                                'action': 'BUY',
                                'shares': shares_to_trade,
                                'price': current_prices[ticker]
                            })
                    else:  # Vente
                        if self.positions[ticker] >= abs(shares_to_trade):
                            self.positions[ticker] += shares_to_trade  # shares_to_trade est négatif
                            self.balance += (cost - commission_cost)
                            self.trades_history.append({
                                'step': self.current_step,
                                'ticker': ticker,
                                'action': 'SELL',
                                'shares': abs(shares_to_trade),
                                'price': current_prices[ticker]
                            })
        
        # Recalculer valeur finale
        positions_value = sum(
            self.positions[ticker] * current_prices[ticker]
            for ticker in self.tickers
        )
        new_portfolio_value = self.balance + positions_value
        
        # Calculer reward (variation en %)
        reward = (new_portfolio_value - self.portfolio_value) / self.portfolio_value
        
        self.portfolio_value = new_portfolio_value
        
        # Terminer si max_steps atteint ou portfolio <= 0
        terminated = (
            self.current_step >= self.data_length - 1 or
            self.portfolio_value <= self.initial_balance * 0.1
        )
        truncated = self.current_step - self.reset_step >= self.max_steps if hasattr(self, 'reset_step') else False
        
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'positions': self.positions.copy(),
            'n_trades': len(self.trades_history)
        }
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self):
        """Construire observation"""
        obs = []
        
        # Features par asset
        for ticker in self.tickers:
            df = self.data[ticker]
            idx = self.current_step
            
            # Prix normalisé
            close = df.iloc[idx]['Close']
            close_norm = (close - df['Close'].iloc[max(0, idx-20):idx].mean()) / df['Close'].iloc[max(0, idx-20):idx].std()
            
            # Volume normalisé
            volume = df.iloc[idx]['Volume']
            volume_norm = (volume - df['Volume'].iloc[max(0, idx-20):idx].mean()) / df['Volume'].iloc[max(0, idx-20):idx].std()
            
            # Returns
            returns_1d = (df.iloc[idx]['Close'] - df.iloc[idx-1]['Close']) / df.iloc[idx-1]['Close'] if idx > 0 else 0
            returns_5d = (df.iloc[idx]['Close'] - df.iloc[idx-5]['Close']) / df.iloc[idx-5]['Close'] if idx > 5 else 0
            
            # RSI simplifié
            gains = df['Close'].diff().clip(lower=0).iloc[max(0, idx-14):idx].mean()
            losses = -df['Close'].diff().clip(upper=0).iloc[max(0, idx-14):idx].mean()
            rsi = 100 - (100 / (1 + gains / (losses + 1e-8)))
            rsi_norm = (rsi - 50) / 50
            
            # MACD approximatif
            ema12 = df['Close'].iloc[max(0, idx-12):idx].ewm(span=12).mean().iloc[-1] if idx > 12 else close
            ema26 = df['Close'].iloc[max(0, idx-26):idx].ewm(span=26).mean().iloc[-1] if idx > 26 else close
            macd = (ema12 - ema26) / close
            
            obs.extend([
                float(close_norm),
                float(volume_norm),
                float(rsi_norm),
                float(macd),
                float(returns_1d),
                float(returns_5d)
            ])
        
        # Portfolio features
        cash_ratio = self.balance / self.portfolio_value
        total_value_norm = (self.portfolio_value - self.initial_balance) / self.initial_balance
        n_positions = sum(1 for pos in self.positions.values() if pos > 0) / self.n_assets
        
        obs.extend([
            float(cash_ratio),
            float(total_value_norm),
            float(n_positions)
        ])
        
        # Remplacer NaN/Inf par 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return np.array(obs, dtype=np.float32)
    
    def render(self, mode='human'):
        """Afficher état"""
        if mode == 'human':
            print(f"\n{'='*60}")
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Cash: ${self.balance:,.2f}")
            print(f"Positions: {self.positions}")
            print(f"Total Trades: {len(self.trades_history)}")
            print(f"{'='*60}")
