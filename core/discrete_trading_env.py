#!/usr/bin/env python3
"""
🎯 ENVIRONNEMENT AVEC ACTIONS DISCRÈTES

SOLUTION FINALE AU PROBLÈME:
Action space continue → L'IA ne découvre jamais SELL

NOUVEAU: ACTIONS DISCRÈTES
- 0 = HOLD (ne rien faire)
- 1 = BUY  (acheter 20% du portfolio disponible)
- 2 = SELL (vendre TOUTES les positions)

Reward:
- BUY → 0 (on attend le résultat)
- SELL → PnL réalisé (profit du trade)
- HOLD + position → 0.5% du PnL latent (encourage à tenir)

Avantages:
1. Signal ultra-clair pour l'IA
2. Pas d'exploration aléatoire nécessaire
3. Force à évaluer explicitement BUY vs SELL
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from collections import deque

class DiscreteTradingEnv(gym.Env):
    """
    Environnement de trading avec actions discrètes
    
    Actions:
    - 0: HOLD  (ne rien faire)
    - 1: BUY   (acheter 20% du portfolio)
    - 2: SELL  (vendre tout)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, prices, initial_balance=100000, commission=0.0001, max_steps=500, buy_pct=0.2):
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
        self.buy_pct = buy_pct  # % du portfolio à investir par BUY
        
        # Pré-calcul returns
        self.returns = np.zeros_like(self.prices)
        self.returns[1:] = (self.prices[1:] - self.prices[:-1]) / (self.prices[:-1] + 1e-8)
        
        # Normalisation prix
        self.prices_mean = np.mean(self.prices)
        self.prices_std = np.std(self.prices) + 1e-8
        self.prices_norm = (self.prices - self.prices_mean) / self.prices_std
        
        # ✅ SPACES DISCRÈTES
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(3,), dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(3)  # 0=HOLD, 1=BUY, 2=SELL
        
        # État
        self.current_step = 50
        self.start_step = 50
        self.balance = initial_balance
        self.shares = 0
        self.portfolio_value = initial_balance
        
        # Tracking
        self.entry_prices = deque()
        self.entry_step = None
        self.total_pnl = 0.0
        self.n_trades = 0
        self.action_counts = [0, 0, 0]  # HOLD, BUY, SELL
    
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
        
        self.entry_prices.clear()
        self.entry_step = None
        self.total_pnl = 0.0
        self.n_trades = 0
        self.action_counts = [0, 0, 0]
        
        return self._get_obs(), {}
    
    def step(self, action):
        self.current_step += 1
        
        if self.current_step >= self.data_length:
            self.current_step = self.data_length - 1
        
        current_price = float(self.prices[self.current_step])
        prev_value = self.balance + self.shares * current_price
        
        # ✅ ACTION DISCRÈTE
        action = int(action)
        self.action_counts[action] += 1
        
        reward = 0.0
        trade_executed = False
        
        # ✅★ ACTION 1: BUY ★
        if action == 1:
            # Acheter buy_pct% du portfolio disponible
            investment = self.balance * self.buy_pct
            
            if investment > 0 and current_price > 0:
                shares_to_buy = int(investment / current_price)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    fee = cost * self.commission
                    total = cost + fee
                    
                    if self.balance >= total:
                        # Exécuter achat
                        self.shares += shares_to_buy
                        self.balance -= total
                        
                        # Enregistrer prix d'entrée
                        for _ in range(shares_to_buy):
                            self.entry_prices.append(current_price)
                        
                        if self.entry_step is None:
                            self.entry_step = self.current_step
                        
                        # ✅ Reward = 0 lors du BUY
                        reward = 0.0
                        trade_executed = True
                        self.n_trades += 1
        
        # ✅★ ACTION 2: SELL ★
        elif action == 2:
            # Vendre TOUTES les positions
            if self.shares > 0 and len(self.entry_prices) > 0:
                proceeds = self.shares * current_price
                fee = proceeds * self.commission
                
                # Calculer PnL réalisé
                pnl_total = 0.0
                shares_sold = self.shares
                
                for _ in range(shares_sold):
                    if len(self.entry_prices) > 0:
                        entry_price = self.entry_prices.popleft()
                        pnl = (current_price - entry_price) / entry_price
                        pnl_total += pnl
                
                avg_pnl = pnl_total / shares_sold if shares_sold > 0 else 0
                
                # ✅ REWARD = PNL RÉALISÉ
                reward = avg_pnl
                
                # Exécuter vente
                self.balance += (proceeds - fee)
                self.shares = 0
                self.entry_step = None
                
                trade_executed = True
                self.total_pnl += avg_pnl
                self.n_trades += 1
        
        # ✅★ ACTION 0: HOLD ★
        else:  # action == 0
            # Ne rien faire, mais récompenser si on tient une position gagnante
            if self.shares > 0 and len(self.entry_prices) > 0:
                avg_entry = np.mean(list(self.entry_prices))
                unrealized_pnl = (current_price - avg_entry) / avg_entry
                
                # ✅ Petit reward sur PnL latent
                reward = unrealized_pnl * 0.005
        
        # Portfolio après action
        new_value = self.balance + self.shares * current_price
        self.portfolio_value = new_value
        
        # Clip reward
        reward = np.clip(reward, -0.3, 0.3)
        
        # Termination
        terminated = (
            self.current_step >= self.data_length - 1 or
            new_value <= self.initial_balance * 0.5
        )
        
        truncated = (self.current_step - self.start_step) >= self.max_steps
        
        # ✅ VENTE FORCÉE À LA FIN
        if (terminated or truncated) and self.shares > 0 and len(self.entry_prices) > 0:
            avg_entry = np.mean(list(self.entry_prices))
            final_pnl = (current_price - avg_entry) / avg_entry
            reward += final_pnl  # Bonus/Pénalité finale
            
            # Vendre (simulation)
            proceeds = self.shares * current_price * (1 - self.commission)
            self.balance += proceeds
            self.shares = 0
            self.portfolio_value = self.balance
        
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'shares': self.shares,
            'trade_executed': trade_executed,
            'total_pnl': self.total_pnl,
            'n_trades': self.n_trades,
            'action_counts': self.action_counts.copy()
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
            print(f"Actions: HOLD={self.action_counts[0]} BUY={self.action_counts[1]} SELL={self.action_counts[2]}")
            if self.shares > 0 and len(self.entry_prices) > 0:
                avg_entry = np.mean(list(self.entry_prices))
                current_price = self.prices[self.current_step]
                unrealized = (current_price - avg_entry) / avg_entry * 100
                print(f"Unrealized PnL: {unrealized:+.2f}%")
            print(f"Total PnL: {self.total_pnl:.4f}")
            print(f"Trades: {self.n_trades}")
