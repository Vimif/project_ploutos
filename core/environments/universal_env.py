"""
UniversalTradingEnv - Environnement universel multi-assets
Version refactorisée et nettoyée
"""

import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict

from .base_env import BaseTradingEnv

class UniversalTradingEnv(BaseTradingEnv):
    """
    Environnement de trading universel
    
    Features:
    - Multi-assets
    - Actions continues (allocation par asset)
    - Gestion commission
    - Observation normalisée
    """
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        initial_balance: float = 100000,
        commission: float = 0.001,
        max_steps: int = 1000,
        lookback_window: int = 50
    ):
        """
        Args:
            data: Données {ticker: DataFrame}
            initial_balance: Capital initial
            commission: Frais transaction
            max_steps: Steps max
            lookback_window: Fenêtre historique pour observation
        """
        super().__init__(data, initial_balance, commission, max_steps)
        
        self.lookback_window = lookback_window
        
        # Vérifier longueur min données
        self.min_length = min(len(df) for df in data.values())
        
        if self.min_length < lookback_window:
            raise ValueError(f"Données trop courtes ({self.min_length} < {lookback_window})")
        
        # Définir espaces
        n_assets = len(self.tickers)
        obs_per_asset = 5  # Close, Volume, Returns, MA, Volatility
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_assets * obs_per_asset + 1,),  # +1 pour balance ratio
            dtype=np.float32
        )
        
        # Actions: allocation pour chaque asset [0, 1]
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(n_assets,),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset avec position aléatoire dans les données"""
        obs, info = super().reset(seed=seed, options=options)
        
        # Démarrer à une position aléatoire (mais pas trop tard)
        if seed is not None:
            np.random.seed(seed)
        
        max_start = self.min_length - self.max_steps - self.lookback_window
        if max_start > self.lookback_window:
            self.current_step = np.random.randint(self.lookback_window, max_start)
        else:
            self.current_step = self.lookback_window
        
        obs = self._get_observation()
        return obs, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Construit observation normalisée
        
        Features par asset:
        - Prix normalisé
        - Volume normalisé
        - Returns
        - Moving average
        - Volatility
        """
        obs = []
        
        for ticker in self.tickers:
            df = self.data[ticker]
            
            # Fenêtre lookback
            start_idx = max(0, self.current_step - self.lookback_window)
            end_idx = self.current_step
            
            window = df.iloc[start_idx:end_idx]
            
            if len(window) == 0:
                # Fallback si pas assez de données
                obs.extend([0.0] * 5)
                continue
            
            # Prix normalisé (% change depuis début fenêtre)
            close_norm = (window['Close'].iloc[-1] / window['Close'].iloc[0]) - 1
            
            # Volume normalisé
            vol_mean = window['Volume'].mean()
            vol_current = window['Volume'].iloc[-1]
            vol_norm = (vol_current / vol_mean) - 1 if vol_mean > 0 else 0
            
            # Returns
            returns = window['Close'].pct_change().iloc[-1]
            
            # Moving average (écart au MA20)
            ma20 = window['Close'].rolling(20, min_periods=1).mean().iloc[-1]
            ma_diff = (window['Close'].iloc[-1] / ma20) - 1
            
            # Volatility (std returns sur fenêtre)
            volatility = window['Close'].pct_change().std()
            
            obs.extend([
                close_norm,
                vol_norm,
                returns if not np.isnan(returns) else 0,
                ma_diff,
                volatility if not np.isnan(volatility) else 0
            ])
        
        # Balance ratio (balance / portfolio_value)
        balance_ratio = self.balance / self.portfolio_value if self.portfolio_value > 0 else 1
        obs.append(balance_ratio)
        
        return np.array(obs, dtype=np.float32)
    
    def _execute_action(self, action):
        """
        Exécute action de trading
        
        Action: array d'allocations [0, 1] par asset
        0 = vendre tout
        1 = investir max possible
        """
        # Normaliser action (somme = 1)
        action = np.array(action)
        action = np.clip(action, 0, 1)
        
        if action.sum() > 0:
            action = action / action.sum()
        
        # Capital disponible pour investissement
        available_capital = self.portfolio_value
        
        # Pour chaque asset
        for i, ticker in enumerate(self.tickers):
            target_allocation = action[i]
            target_value = available_capital * target_allocation
            
            current_price = self._get_current_price(ticker)
            
            if current_price <= 0:
                continue
            
            # Shares actuelles
            current_shares = self.positions[ticker]
            current_value = current_shares * current_price
            
            # Shares cibles
            target_shares = int(target_value / current_price)
            
            # Différence
            shares_diff = target_shares - current_shares
            
            if shares_diff > 0:
                # Acheter
                cost = shares_diff * current_price * (1 + self.commission)
                
                if cost <= self.balance:
                    self.positions[ticker] = target_shares
                    self.balance -= cost
                    
            elif shares_diff < 0:
                # Vendre
                proceeds = abs(shares_diff) * current_price * (1 - self.commission)
                self.positions[ticker] = target_shares
                self.balance += proceeds
    
    def _calculate_reward(self, action, obs, info) -> float:
        """
        Reward basé sur profit/loss
        
        Encourage:
        - Gains (reward positif)
        - Pénalise pertes (reward négatif)
        """
        if len(self.portfolio_history) < 2:
            return 0.0
        
        prev_value = self.portfolio_history[-2]
        current_value = self.portfolio_history[-1]
        
        # Reward = % change
        reward = (current_value - prev_value) / prev_value
        
        # Multiplier pour scale
        reward *= 100
        
        return float(reward)
    
    def _is_terminated(self) -> bool:
        """Terminé si faillite"""
        return self.portfolio_value <= self.initial_balance * 0.1  # -90%
