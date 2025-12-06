#!/usr/bin/env python3
"""
🎯 ENVIRONNEMENT AVEC REWARD SUR PNL RÉALISÉ

SOLUTION AU PROBLÈME FONDAMENTAL:
L'ancien reward = (new_portfolio - prev_portfolio) / prev_portfolio
→ Toujours négatif à cause des frais
→ L'IA apprend à ne rien faire

NOUVEAU REWARD = PnL du trade
- BUY @ $530 puis SELL @ $550 → Reward = +0.0377 (+3.77%) ✅
- BUY @ $550 puis SELL @ $530 → Reward = -0.0364 (-3.64%) ❌
- HOLD (pas de trade) → Reward = 0

Avantages:
1. Signal CLAIR: Acheter bas, vendre haut = positif
2. Les frais sont un coût réel mais n'impactent pas le reward
3. L'IA comprend la QUALITÉ de ses décisions
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from collections import deque

class SimplePnLTradingEnv(gym.Env):
    """
    Environnement qui récompense le PnL réalisé
    
    Différences clés:
    - Tracking des prix d'entrée pour chaque position
    - Reward = 0 lors du BUY (on attend le résultat)
    - Reward = PnL lors du SELL (performance du trade)
    - Reward = 0 si HOLD (encourage sélectivité)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, prices, initial_balance=100000, commission=0.0001, max_steps=500):
        super().__init__()
        
        # Prix
        if isinstance(prices, pd.Series):
            self.prices = prices.values.flatten()
        else:
            self.prices = np.array(prices, dtype=np.float32).flatten()
        
        self.data_length = len(self.prices)
        self.initial_balance = initial_balance
        self.commission = commission
        self.max_steps = max_steps
        
        # Pré-calcul returns
        self.returns = np.zeros_like(self.prices)
        self.returns[1:] = (self.prices[1:] - self.prices[:-1]) / (self.prices[:-1] + 1e-8)
        
        # Normalisation prix
        self.prices_mean = np.mean(self.prices)
        self.prices_std = np.std(self.prices) + 1e-8
        self.prices_norm = (self.prices - self.prices_mean) / self.prices_std
        
        # Spaces
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(3,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )
        
        # État
        self.current_step = 50
        self.start_step = 50
        self.balance = initial_balance
        self.shares = 0
        self.portfolio_value = initial_balance
        
        # ✅ NOUVEAU: Tracking pour PnL
        self.entry_prices = deque()  # Prix d'entrée pour chaque share
        self.total_pnl = 0.0
        self.n_trades = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.data_length > self.max_steps + 100:
            self.current_step = np.random.randint(50, self.data_length - self.max_steps - 1)
        else:
            self.current_step = 50
        
        self.start_step = self.current_step
        self.balance = self.initial_balance
        self.shares = 0
        self.portfolio_value = self.initial_balance
        
        # Reset tracking
        self.entry_prices.clear()
        self.total_pnl = 0.0
        self.n_trades = 0
        
        return self._get_obs(), {}
    
    def step(self, action):
        self.current_step += 1
        
        if self.current_step >= self.data_length:
            self.current_step = self.data_length - 1
        
        current_price = float(self.prices[self.current_step])
        prev_value = self.balance + self.shares * current_price
        
        action_val = float(np.clip(action[0], -1, 1))
        
        # Target investment
        target_investment = max(0, action_val) * prev_value
        current_investment = self.shares * current_price
        trade_amount = target_investment - current_investment
        
        # ✅ NOUVEAU: Reward par défaut = 0 (encourage sélectivité)
        reward = 0.0
        trade_executed = False
        
        # Exécuter trade si significatif
        if abs(trade_amount) > prev_value * 0.02:  # ✅ Seuil augmenté à 2%
            shares_to_trade = int(trade_amount / current_price)
            
            if shares_to_trade > 0:  # BUY
                cost = shares_to_trade * current_price
                fee = cost * self.commission
                total = cost + fee
                
                if self.balance >= total:
                    # Exécuter achat
                    self.shares += shares_to_trade
                    self.balance -= total
                    
                    # ✅ Enregistrer prix d'entrée pour chaque share
                    for _ in range(shares_to_trade):
                        self.entry_prices.append(current_price)
                    
                    # ✅ Reward = 0 lors de l'achat (on attend le résultat)
                    reward = 0.0
                    trade_executed = True
                    self.n_trades += 1
            
            elif shares_to_trade < 0:  # SELL
                shares_to_sell = min(abs(shares_to_trade), self.shares)
                
                if shares_to_sell > 0 and len(self.entry_prices) > 0:
                    proceeds = shares_to_sell * current_price
                    fee = proceeds * self.commission
                    
                    # ✅ CALCULER LE PNL RÉALISÉ
                    pnl_total = 0.0
                    for _ in range(shares_to_sell):
                        if len(self.entry_prices) > 0:
                            entry_price = self.entry_prices.popleft()
                            pnl = (current_price - entry_price) / entry_price
                            pnl_total += pnl
                    
                    # Moyenne du PnL sur les shares vendues
                    avg_pnl = pnl_total / shares_to_sell if shares_to_sell > 0 else 0
                    
                    # ✅ REWARD = PNL MOYEN DU TRADE
                    reward = avg_pnl
                    
                    # Exécuter vente
                    self.shares -= shares_to_sell
                    self.balance += (proceeds - fee)
                    
                    trade_executed = True
                    self.total_pnl += avg_pnl
                    self.n_trades += 1
        
        # Portfolio après trade
        new_value = self.balance + self.shares * current_price
        self.portfolio_value = new_value
        
        # ✅ Clip reward pour stabilité
        reward = np.clip(reward, -0.2, 0.2)
        
        # Termination
        terminated = (
            self.current_step >= self.data_length - 1 or
            new_value <= self.initial_balance * 0.5
        )
        
        truncated = (self.current_step - self.start_step) >= self.max_steps
        
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'shares': self.shares,
            'trade_executed': trade_executed,
            'total_pnl': self.total_pnl,
            'n_trades': self.n_trades
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        """Observation: [prix_norm, returns, cash_ratio]"""
        idx = self.current_step
        
        price_norm = float(self.prices_norm[idx])
        returns = float(self.returns[idx])
        cash_ratio = float(self.balance / self.portfolio_value) if self.portfolio_value > 0 else 1.0
        
        obs = np.array([price_norm, returns, cash_ratio], dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        obs = np.clip(obs, -10, 10)
        
        return obs
    
    def render(self, mode='human'):
        if mode == 'human':
            print(f"\nStep: {self.current_step}")
            print(f"Portfolio: ${self.portfolio_value:,.2f}")
            print(f"Balance: ${self.balance:,.2f}")
            print(f"Shares: {self.shares}")
            print(f"Total PnL: {self.total_pnl:.4f}")
            print(f"Trades: {self.n_trades}")
