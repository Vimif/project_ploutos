#!/usr/bin/env python3
"""
🔍 SCRIPT DEBUG VERBOSE - TRACE COMPLÈTE

Objectif: Identifier POURQUOI l'IA ne fait rien

Problème observé:
- Portfolio reste à $100,000 exact
- 100% actions SELL
- Sharpe = 0.000

Hypothèses:
1. Observations toujours identiques (l'IA est "aveugle")
2. Rewards toujours négatifs ou nuls
3. Trades jamais exécutés (bug arrondi ou logique)

Ce script va PRINT les 20 premiers steps pour voir exactement ce qui se passe.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
import yfinance as yf
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🔍 DEBUG VERBOSE - TRACE STEP BY STEP")
print("="*80)

# ============================================================================
# ENVIRONNEMENT AVEC PRINTS DEBUG
# ============================================================================

class VerboseDebugEnv(gym.Env):
    """Environnement qui PRINT tout pour debug"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, prices, initial_balance=100000, commission=0.0001, max_steps=500, verbose=False):
        super().__init__()
        
        if isinstance(prices, pd.Series):
            self.prices = prices.values.flatten()
        else:
            self.prices = np.array(prices, dtype=np.float32).flatten()
        
        self.data_length = len(self.prices)
        self.initial_balance = initial_balance
        self.commission = commission
        self.max_steps = max_steps
        self.verbose = verbose
        
        # Pré-calcul
        self.returns = np.zeros_like(self.prices)
        self.returns[1:] = (self.prices[1:] - self.prices[:-1]) / (self.prices[:-1] + 1e-8)
        
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
        self.step_count = 0
    
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
        self.step_count = 0
        
        obs = self._get_obs()
        
        if self.verbose:
            print("\n" + "="*60)
            print("🔄 RESET")
            print(f"   Start step: {self.current_step}")
            print(f"   Price: ${self.prices[self.current_step]:.2f}")
            print(f"   Obs: {obs}")
            print("="*60)
        
        return obs, {}
    
    def step(self, action):
        self.current_step += 1
        self.step_count += 1
        
        if self.current_step >= self.data_length:
            self.current_step = self.data_length - 1
        
        current_price = float(self.prices[self.current_step])
        
        # Portfolio AVANT
        prev_value = self.balance + self.shares * current_price
        
        # Action
        action_val = float(np.clip(action[0], -1, 1))
        
        # 🚨 VERBOSE: Print les 20 premiers steps
        if self.verbose and self.step_count <= 20:
            print(f"\n--- STEP {self.step_count} ---")
            print(f"Price: ${current_price:.2f}")
            print(f"Balance: ${self.balance:,.2f} | Shares: {self.shares}")
            print(f"Portfolio: ${prev_value:,.2f}")
            print(f"Action raw: {action_val:.4f}")
        
        # Target
        target_investment = max(0, action_val) * prev_value
        current_investment = self.shares * current_price
        trade_amount = target_investment - current_investment
        
        if self.verbose and self.step_count <= 20:
            print(f"Target invest: ${target_investment:,.2f}")
            print(f"Current invest: ${current_investment:,.2f}")
            print(f"Trade amount: ${trade_amount:,.2f}")
        
        # Exécuter trade
        trade_executed = False
        
        if abs(trade_amount) > prev_value * 0.01:
            shares_to_trade = int(trade_amount / current_price)
            
            if self.verbose and self.step_count <= 20:
                print(f"Shares to trade: {shares_to_trade}")
            
            if shares_to_trade > 0:  # BUY
                cost = shares_to_trade * current_price
                fee = cost * self.commission
                total = cost + fee
                
                if self.balance >= total:
                    self.shares += shares_to_trade
                    self.balance -= total
                    trade_executed = True
                    
                    if self.verbose and self.step_count <= 20:
                        print(f"✅ BUY executed: {shares_to_trade} shares @ ${current_price:.2f}")
                        print(f"   Cost: ${total:,.2f} (fee: ${fee:.2f})")
                else:
                    if self.verbose and self.step_count <= 20:
                        print(f"❌ BUY rejected: Insufficient balance")
            
            elif shares_to_trade < 0:  # SELL
                shares_to_sell = min(abs(shares_to_trade), self.shares)
                
                if shares_to_sell > 0:
                    proceeds = shares_to_sell * current_price
                    fee = proceeds * self.commission
                    
                    self.shares -= shares_to_sell
                    self.balance += (proceeds - fee)
                    trade_executed = True
                    
                    if self.verbose and self.step_count <= 20:
                        print(f"✅ SELL executed: {shares_to_sell} shares @ ${current_price:.2f}")
                        print(f"   Proceeds: ${proceeds - fee:,.2f} (fee: ${fee:.2f})")
                else:
                    if self.verbose and self.step_count <= 20:
                        print(f"❌ SELL rejected: No shares to sell")
        else:
            if self.verbose and self.step_count <= 20:
                print(f"⏸️  No trade (amount < 1% portfolio)")
        
        # Portfolio APRÈS
        new_value = self.balance + self.shares * current_price
        self.portfolio_value = new_value
        
        # Reward
        if prev_value > 0:
            reward = (new_value - prev_value) / prev_value
        else:
            reward = -1.0
        
        reward = np.clip(reward, -0.1, 0.1)
        
        if self.verbose and self.step_count <= 20:
            print(f"New portfolio: ${new_value:,.2f}")
            print(f"Reward: {reward:.6f}")
        
        # Termination
        terminated = (
            self.current_step >= self.data_length - 1 or
            new_value <= self.initial_balance * 0.3
        )
        
        truncated = (self.current_step - self.start_step) >= self.max_steps
        
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'shares': self.shares,
            'trade_executed': trade_executed
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        idx = self.current_step
        
        price_norm = float(self.prices_norm[idx])
        returns = float(self.returns[idx])
        cash_ratio = float(self.balance / self.portfolio_value) if self.portfolio_value > 0 else 1.0
        
        obs = np.array([price_norm, returns, cash_ratio], dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        obs = np.clip(obs, -10, 10)
        
        return obs

# ============================================================================
# TEST VERBOSE
# ============================================================================

def test_verbose():
    print("\n📡 Chargement données SPY...")
    
    end = datetime.now()
    start = end - timedelta(days=600)
    
    df = yf.download('SPY', start=start, end=end, interval='1d', progress=False)
    df = df.tail(min(500, len(df)))
    prices = df['Close'].values
    
    print(f"   ✅ {len(prices)} jours chargés")
    print(f"   Prix moyen: ${np.mean(prices):.2f}")
    print(f"   Prix std: ${np.std(prices):.2f}\n")
    
    # Créer env VERBOSE
    print("🔍 Création env verbose...")
    env = VerboseDebugEnv(prices, verbose=True)
    
    # Créer modèle simple
    print("🧠 Initialisation PPO...\n")
    vec_env = DummyVecEnv([lambda: env])
    
    model = PPO(
        'MlpPolicy',
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        verbose=0
    )
    
    print("🚀 Entraînement rapide (50k steps)...\n")
    print("="*80)
    print("OBSERVATION DES 20 PREMIERS STEPS")
    print("="*80)
    
    model.learn(total_timesteps=50_000, progress_bar=False)
    
    print("\n" + "="*80)
    print("🎯 ANALYSE")
    print("="*80)
    
    print("\nVérifications:")
    print("1. Les observations varient-elles ? (prix_norm doit changer)")
    print("2. Les rewards sont-ils calculés ? (doivent être non nuls)")
    print("3. Les trades s'exécutent-ils ? (BUY/SELL doit apparaître)")
    print("\nSi RIEN ne s'exécute → Bug dans la logique step()")
    print("Si observations = 0 → Bug normalisation")
    print("Si rewards = 0 → L'IA n'apprend rien\n")
    print("="*80)

if __name__ == '__main__':
    test_verbose()
