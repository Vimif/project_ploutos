#!/usr/bin/env python3
"""
🧪 SCRIPT DE DEBUG - ENVIRONNEMENT MINIMAL

Test de santé pour isoler le problème:
- Si ce script RÉUSSIT → Le problème vient de la complexité (reward, observations, curriculum)
- Si ce script ÉCHOUE → Bug fondamental dans l'environnement de base

Environnement Ultra-Simple:
  - 1 asset: SPY
  - 500 derniers jours uniquement
  - Observations: [Prix normalisé, Returns, Cash Ratio]
  - Reward: Profit pur (pas de Sharpe/Drawdown)
  - Commission: 0.01%
  - Pas de realistic costs

Critères de Succès:
  ✅ Portfolio > $102,000 (+2%)
  ✅ Sharpe > 0.5
  ✅ Buy% > 10% ET Sell% > 10% (pas bloqué en HOLD/SELL)
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
print("🧪 DEBUG ENVIRONNEMENT MINIMAL")
print("="*80)

# ============================================================================
# CLASSE ENVIRONNEMENT SIMPLE
# ============================================================================

class SimpleDebugEnv(gym.Env):
    """
    Environnement ULTRA-MINIMALISTE pour diagnostic
    
    Caractéristiques:
    - 1 seul asset
    - Observations: 3 features seulement
    - Reward: Profit brut
    - Logique step() simplifiée
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, prices, initial_balance=100000, commission=0.0001, max_steps=500):
        super().__init__()
        
        self.prices = np.array(prices, dtype=np.float32)
        self.data_length = len(self.prices)
        self.initial_balance = initial_balance
        self.commission = commission
        self.max_steps = max_steps
        
        # Pré-calcul returns
        self.returns = np.zeros_like(self.prices)
        self.returns[1:] = (self.prices[1:] - self.prices[:-1]) / self.prices[:-1]
        
        # Normalisation prix (z-score)
        self.prices_mean = np.mean(self.prices)
        self.prices_std = np.std(self.prices)
        self.prices_norm = (self.prices - self.prices_mean) / (self.prices_std + 1e-8)
        
        # Spaces
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(3,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Départ aléatoire
        if self.data_length > self.max_steps + 100:
            self.current_step = np.random.randint(50, self.data_length - self.max_steps - 1)
        else:
            self.current_step = 50
        
        self.start_step = self.current_step
        self.balance = self.initial_balance
        self.shares = 0
        self.portfolio_value = self.initial_balance
        
        return self._get_obs(), {}
    
    def step(self, action):
        self.current_step += 1
        
        if self.current_step >= self.data_length:
            self.current_step = self.data_length - 1
        
        current_price = float(self.prices[self.current_step])
        
        # Portfolio avant trade
        prev_value = self.balance + self.shares * current_price
        
        # Interpréter action: [-1, 1] → pourcentage à investir
        action_val = float(np.clip(action[0], -1, 1))
        
        # Seulement investir si action > 0
        target_investment = max(0, action_val) * prev_value
        current_investment = self.shares * current_price
        
        trade_amount = target_investment - current_investment
        
        # Exécuter si significatif (> 1% portfolio)
        if abs(trade_amount) > prev_value * 0.01:
            shares_to_trade = int(trade_amount / current_price)
            
            if shares_to_trade > 0:  # BUY
                cost = shares_to_trade * current_price
                fee = cost * self.commission
                total = cost + fee
                
                if self.balance >= total:
                    self.shares += shares_to_trade
                    self.balance -= total
            
            elif shares_to_trade < 0:  # SELL
                shares_to_sell = min(abs(shares_to_trade), self.shares)
                
                if shares_to_sell > 0:
                    proceeds = shares_to_sell * current_price
                    fee = proceeds * self.commission
                    
                    self.shares -= shares_to_sell
                    self.balance += (proceeds - fee)
        
        # Portfolio après trade
        new_value = self.balance + self.shares * current_price
        self.portfolio_value = new_value
        
        # REWARD = Profit normalisé
        if prev_value > 0:
            reward = (new_value - prev_value) / prev_value
        else:
            reward = -1.0
        
        reward = np.clip(reward, -0.1, 0.1)
        
        # Termination
        terminated = (
            self.current_step >= self.data_length - 1 or
            new_value <= self.initial_balance * 0.3
        )
        
        truncated = (self.current_step - self.start_step) >= self.max_steps
        
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'shares': self.shares
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        """Obs: [prix_norm, returns, cash_ratio]"""
        idx = self.current_step
        
        price_norm = self.prices_norm[idx]
        returns = self.returns[idx]
        cash_ratio = self.balance / self.portfolio_value if self.portfolio_value > 0 else 1.0
        
        obs = np.array([price_norm, returns, cash_ratio], dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        obs = np.clip(obs, -10, 10)
        
        return obs

# ============================================================================
# ÉVALUATION
# ============================================================================

def evaluate_debug(model, env, n_episodes=20):
    """Évalue le modèle"""
    portfolios = []
    actions = []
    
    for _ in range(n_episodes):
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        
        done = False
        ep_values = [100000]
        ep_actions = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            ep_actions.append(float(action[0]))
            
            result = env.step(action)
            
            if len(result) == 5:
                obs, reward, term, trunc, info = result
                done = term or trunc
            else:
                obs, reward, done, info = result
            
            ep_values.append(info['portfolio_value'])
        
        portfolios.append(ep_values[-1])
        actions.extend(ep_actions)
    
    # Métriques
    mean_pf = np.mean(portfolios)
    
    # Sharpe
    returns = [(p - 100000) / 100000 for p in portfolios]
    sharpe = 0
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        sharpe = np.clip(sharpe, -10, 10)
    
    # Actions
    actions_arr = np.array(actions)
    buy_pct = (np.sum(actions_arr > 0.33) / len(actions_arr)) * 100
    hold_pct = (np.sum(np.abs(actions_arr) <= 0.33) / len(actions_arr)) * 100
    sell_pct = (np.sum(actions_arr < -0.33) / len(actions_arr)) * 100
    
    return {
        'mean_portfolio': mean_pf,
        'min_portfolio': np.min(portfolios),
        'max_portfolio': np.max(portfolios),
        'sharpe': sharpe,
        'buy_pct': buy_pct,
        'hold_pct': hold_pct,
        'sell_pct': sell_pct
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n📡 1. Chargement données SPY (500 derniers jours)...")
    
    end = datetime.now()
    start = end - timedelta(days=600)  # Marge pour jours non-trading
    
    df = yf.download('SPY', start=start, end=end, interval='1d', progress=False)
    
    if df.empty or len(df) < 100:
        print("❌ Erreur chargement données")
        return False
    
    # Garder 500 derniers jours
    df = df.tail(500)
    prices = df['Close'].values
    
    print(f"   ✅ {len(prices)} jours chargés\n")
    
    # Créer env
    print("🏗️  2. Création environnement debug...")
    env = SimpleDebugEnv(prices, initial_balance=100000, commission=0.0001, max_steps=400)
    vec_env = DummyVecEnv([lambda: env])
    print("   ✅ Env créé\n")
    
    # Créer modèle
    print("🧠 3. Initialisation PPO...")
    model = PPO(
        'MlpPolicy',
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={'net_arch': [128, 128]},
        verbose=0
    )
    print("   ✅ Modèle initialisé\n")
    
    # Entraînement
    print("🚀 4. Entraînement (200k steps, ~15min)...")
    print("   Objectif: Overfit sur ces 500 jours\n")
    
    try:
        model.learn(total_timesteps=200_000, progress_bar=True)
    except KeyboardInterrupt:
        print("\n⚠️  Interrompu")
    
    print("\n   ✅ Entraînement terminé\n")
    
    # Évaluation
    print("📊 5. Évaluation finale...")
    metrics = evaluate_debug(model, env, n_episodes=20)
    
    print("\n" + "="*80)
    print("🎯 RÉSULTATS TEST DEBUG")
    print("="*80)
    
    profit_pct = (metrics['mean_portfolio'] - 100000) / 1000
    
    print(f"\n💰 PORTFOLIO:")
    print(f"   Moyen : ${metrics['mean_portfolio']:,.0f} ({profit_pct:+.1f}%)")
    print(f"   Min   : ${metrics['min_portfolio']:,.0f}")
    print(f"   Max   : ${metrics['max_portfolio']:,.0f}")
    
    print(f"\n📈 MÉTRIQUES:")
    print(f"   Sharpe: {metrics['sharpe']:.3f}")
    
    print(f"\n🎯 ACTIONS:")
    print(f"   BUY   : {metrics['buy_pct']:5.1f}%")
    print(f"   HOLD  : {metrics['hold_pct']:5.1f}%")
    print(f"   SELL  : {metrics['sell_pct']:5.1f}%")
    
    # Validation
    print("\n" + "="*80)
    print("✅ VALIDATION")
    print("="*80)
    
    success = True
    
    if metrics['mean_portfolio'] >= 102000:
        print("\n✅ Portfolio > $102k : PASS")
    else:
        print(f"\n❌ Portfolio trop faible : ${metrics['mean_portfolio']:,.0f} < $102k")
        success = False
    
    if metrics['sharpe'] >= 0.5:
        print("✅ Sharpe > 0.5 : PASS")
    else:
        print(f"❌ Sharpe trop faible : {metrics['sharpe']:.3f} < 0.5")
        success = False
    
    if metrics['buy_pct'] >= 10 and metrics['sell_pct'] >= 10:
        print("✅ Actions équilibrées : PASS")
    else:
        print(f"❌ Actions déséquilibrées : BUY {metrics['buy_pct']:.1f}% / SELL {metrics['sell_pct']:.1f}%")
        success = False
    
    print("\n" + "="*80)
    
    if success:
        print("✅ TEST DEBUG : SUCCÈS")
        print("\n👉 Conclusion: L'algorithme PPO FONCTIONNE.")
        print("   Le problème vient de la COMPLEXITÉ.")
        print("\n🛠️  Prochaines étapes:")
        print("   1. Réintégrer complexité PIÈCE PAR PIÈCE")
        print("   2. Tester avec RSI/MACD (une à la fois)")
        print("   3. Tester avec MultiTaskReward")
        print("   4. Identifier quelle pièce casse le système")
    else:
        print("❌ TEST DEBUG : ÉCHEC")
        print("\n👉 Conclusion: PROBLÈME FONDAMENTAL.")
        print("   Même en cas simple, l'IA échoue.")
        print("\n🔍 Investigations:")
        print("   1. Vérifier logique step() en détail")
        print("   2. Vérifier calcul reward (print dans la boucle)")
        print("   3. Tester avec actions discrètes")
        print("   4. Comparer avec gym example simple")
    
    print("="*80)
    
    return success

if __name__ == '__main__':
    print("\n🎯 Lance test debug...\n")
    
    success = main()
    
    exit(0 if success else 1)
