"""
Environnement de base pour tous les environnements Ploutos
Définit l'interface commune
"""

import gymnasium as gym
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd

class BaseTradingEnv(gym.Env, ABC):
    """
    Classe de base pour environnements de trading
    
    Tous les environnements Ploutos doivent hériter de cette classe
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        initial_balance: float = 100000,
        commission: float = 0.001,
        max_steps: int = 1000
    ):
        """
        Args:
            data: Données {ticker: DataFrame}
            initial_balance: Capital initial
            commission: Frais de transaction (0.001 = 0.1%)
            max_steps: Steps maximum par épisode
        """
        super().__init__()
        
        self.data = data
        self.tickers = list(data.keys())
        self.initial_balance = initial_balance
        self.commission = commission
        self.max_steps = max_steps
        
        # État
        self.current_step = 0
        self.balance = initial_balance
        self.positions = {ticker: 0 for ticker in self.tickers}
        self.portfolio_value = initial_balance
        
        # Historique
        self.portfolio_history = []
        
        # Espaces (à définir dans enfants)
        self.observation_space = None
        self.action_space = None
    
    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        """Retourne l'observation courante"""
        pass
    
    @abstractmethod
    def _calculate_reward(self, action, obs, info) -> float:
        """Calcule la récompense"""
        pass
    
    @abstractmethod
    def _execute_action(self, action):
        """Exécute l'action de trading"""
        pass
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environnement"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.positions = {ticker: 0 for ticker in self.tickers}
        self.portfolio_value = self.initial_balance
        self.portfolio_history = [self.initial_balance]
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Exécute un step"""
        
        # Sauvegarder état précédent
        prev_value = self.portfolio_value
        
        # Exécuter action
        self._execute_action(action)
        
        # Avancer temps
        self.current_step += 1
        
        # Calculer valeur portfolio
        self._update_portfolio_value()
        
        # Observation
        obs = self._get_observation()
        
        # Info
        info = self._get_info()
        
        # Reward
        reward = self._calculate_reward(action, obs, info)
        
        # Terminated / Truncated
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        
        # Historique
        self.portfolio_history.append(self.portfolio_value)
        
        return obs, reward, terminated, truncated, info
    
    def _update_portfolio_value(self):
        """Recalcule valeur totale du portfolio"""
        positions_value = 0
        
        for ticker, shares in self.positions.items():
            if shares > 0:
                current_price = self._get_current_price(ticker)
                positions_value += shares * current_price
        
        self.portfolio_value = self.balance + positions_value
    
    def _get_current_price(self, ticker: str) -> float:
        """Prix actuel d'un ticker"""
        if self.current_step >= len(self.data[ticker]):
            return self.data[ticker]['Close'].iloc[-1]
        
        return self.data[ticker]['Close'].iloc[self.current_step]
    
    def _is_terminated(self) -> bool:
        """Episode terminé (bankrupt ou autre condition)"""
        return self.portfolio_value <= 0
    
    def _get_info(self) -> Dict[str, Any]:
        """Informations additionnelles"""
        return {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'positions': self.positions.copy(),
            'step': self.current_step
        }
    
    def render(self):
        """Affichage (optionnel)"""
        pass
