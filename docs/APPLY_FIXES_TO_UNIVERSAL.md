# 🛠️ GUIDE: APPLIQUER LES FIXES À UNIVERSALTRADINGENV

Ce guide explique comment appliquer les solutions validées dans `DiscreteTradingEnv` au système complet `UniversalTradingEnv`.

---

## 🎯 OBJECTIF

Adapter le système validé (1 asset, actions discrètes, reward PnL) au trading multi-assets avec toutes les features.

---

## 📑 ÉTAPE 1: MODIFIER ACTION SPACE

### Fichier: `core/universal_trading_environment.py`

#### Code actuel (AVANT)
```python
self.action_space = spaces.Box(
    low=-1, 
    high=1, 
    shape=(len(self.tickers),), 
    dtype=np.float32
)
```

#### Code modifié (APRÈS)
```python
from gymnasium.spaces import MultiDiscrete

self.action_space = MultiDiscrete([3] * len(self.tickers))
# [3, 3, 3, ...] = 3 actions possibles par ticker
# 0 = HOLD
# 1 = BUY
# 2 = SELL
```

---

## 📒 ÉTAPE 2: AJOUTER TRACKING PNL

### Ajouter dans `__init__`

```python
from collections import deque

class UniversalTradingEnv:
    def __init__(self, ...):
        # ... code existant ...
        
        # ✅ NOUVEAU: Tracking PnL par ticker
        self.entry_prices = {
            ticker: deque() for ticker in self.tickers
        }
        self.entry_steps = {
            ticker: None for ticker in self.tickers
        }
```

### Ajouter dans `reset()`

```python
def reset(self, seed=None, options=None):
    # ... code existant ...
    
    # ✅ NOUVEAU: Reset tracking
    for ticker in self.tickers:
        self.entry_prices[ticker].clear()
        self.entry_steps[ticker] = None
    
    return self._get_obs(), {}
```

---

## 📓 ÉTAPE 3: RÉÉCRIRE LA LOGIQUE STEP()

### Structure générale

```python
def step(self, actions):
    # actions = [0, 1, 2, 0, 1]  # Un int par ticker
    
    self.current_step += 1
    total_reward = 0.0
    
    # Pour chaque ticker
    for i, ticker in enumerate(self.tickers):
        action = int(actions[i])
        current_price = self.prices[ticker][self.current_step]
        
        # Calculer reward pour ce ticker
        reward_ticker = self._execute_action(
            ticker, action, current_price
        )
        total_reward += reward_ticker
    
    # ... reste du code (termination, info, etc.) ...
    
    return obs, total_reward, terminated, truncated, info
```

### Méthode `_execute_action`

```python
def _execute_action(self, ticker, action, current_price):
    """
    Exécute une action pour un ticker et retourne le reward.
    
    Args:
        ticker: Nom du ticker (ex: 'NVDA')
        action: 0=HOLD, 1=BUY, 2=SELL
        current_price: Prix actuel du ticker
    
    Returns:
        reward (float)
    """
    reward = 0.0
    
    # ★ ACTION 1: BUY ★
    if action == 1:
        # Investir 20% du portfolio dans ce ticker
        investment = self.balance * 0.2
        
        if investment > 0 and current_price > 0:
            shares_to_buy = int(investment / current_price)
            
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                fee = cost * self.commission
                total = cost + fee
                
                if self.balance >= total:
                    # Exécuter achat
                    self.shares[ticker] += shares_to_buy
                    self.balance -= total
                    
                    # Enregistrer prix d'entrée
                    for _ in range(shares_to_buy):
                        self.entry_prices[ticker].append(current_price)
                    
                    if self.entry_steps[ticker] is None:
                        self.entry_steps[ticker] = self.current_step
                    
                    # Reward = 0 lors du BUY
                    reward = 0.0
    
    # ★ ACTION 2: SELL ★
    elif action == 2:
        if self.shares[ticker] > 0 and len(self.entry_prices[ticker]) > 0:
            shares_to_sell = self.shares[ticker]
            proceeds = shares_to_sell * current_price
            fee = proceeds * self.commission
            
            # Calculer PnL réalisé
            pnl_total = 0.0
            for _ in range(shares_to_sell):
                if len(self.entry_prices[ticker]) > 0:
                    entry = self.entry_prices[ticker].popleft()
                    pnl = (current_price - entry) / entry
                    pnl_total += pnl
            
            avg_pnl = pnl_total / shares_to_sell
            
            # ✅ REWARD = PNL RÉALISÉ
            reward = avg_pnl
            
            # Exécuter vente
            self.balance += (proceeds - fee)
            self.shares[ticker] = 0
            self.entry_steps[ticker] = None
    
    # ★ ACTION 0: HOLD ★
    else:
        # Reward sur PnL latent si on tient une position
        if self.shares[ticker] > 0 and len(self.entry_prices[ticker]) > 0:
            avg_entry = np.mean(list(self.entry_prices[ticker]))
            unrealized_pnl = (current_price - avg_entry) / avg_entry
            reward = unrealized_pnl * 0.005  # 0.5% du PnL latent
    
    # Clip reward pour stabilité
    reward = np.clip(reward, -0.3, 0.3)
    
    return reward
```

---

## 📔 ÉTAPE 4: VENTE FORCÉE À LA FIN

### Ajouter dans `step()` avant le return

```python
def step(self, actions):
    # ... code existant ...
    
    # Termination
    terminated = (...)
    truncated = (...)
    
    # ✅ VENTE FORCÉE À LA FIN
    if terminated or truncated:
        for ticker in self.tickers:
            if self.shares[ticker] > 0 and len(self.entry_prices[ticker]) > 0:
                # Calculer PnL final
                current_price = self.prices[ticker][self.current_step]
                avg_entry = np.mean(list(self.entry_prices[ticker]))
                final_pnl = (current_price - avg_entry) / avg_entry
                
                # Ajouter au reward
                total_reward += final_pnl
                
                # Vendre (simulation)
                proceeds = self.shares[ticker] * current_price
                proceeds *= (1 - self.commission)
                self.balance += proceeds
                self.shares[ticker] = 0
    
    return obs, total_reward, terminated, truncated, info
```

---

## 📕 ÉTAPE 5: CRÉER SCRIPT DE TEST

### Fichier: `scripts/test_universal_discrete.py`

```python
#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.universal_trading_environment import UniversalTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Créer env avec actions discrètes
tickers = ['NVDA', 'MSFT', 'AAPL', 'SPY', 'QQQ']

env = UniversalTradingEnv(
    tickers=tickers,
    start_date='2023-01-01',
    end_date='2024-12-31',
    initial_balance=100000,
    commission=0.0001,
    # ... autres params ...
)

vec_env = DummyVecEnv([lambda: env])

# Modèle PPO
model = PPO(
    'MlpPolicy',
    vec_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.05,  # Exploration
    verbose=1
)

# Entraînement
model.learn(total_timesteps=500_000)

# Sauvegarder
model.save('models/universal_discrete_v1')

print("✅ Modèle entraîné et sauvegardé!")
```

---

## ✅ CHECKLIST DE VALIDATION

### Avant de lancer l'entraînement

- [ ] Action space = `MultiDiscrete([3, 3, 3, ...])`
- [ ] Tracking `entry_prices` par ticker implémenté
- [ ] Logique BUY/HOLD/SELL par ticker implémentée
- [ ] Reward = PnL réalisé + 0.5% PnL latent
- [ ] Vente forcée à la fin implémentée
- [ ] Script de test créé

### Après 100k steps

- [ ] L'IA utilise BUY (> 10% des actions)
- [ ] L'IA utilise SELL (> 10% des actions)
- [ ] Portfolio > $102k
- [ ] Pas de crash/erreur

### Après 500k steps

- [ ] Sharpe > 0.5
- [ ] Actions équilibrées sur tous les tickers
- [ ] Variance entre épisodes > $1000

---

## 🚨 PIÈGES À ÉVITER

### 1. Oublier le tracking par ticker
```python
# ❌ MAUVAIS
self.entry_prices = deque()  # Global

# ✅ BON
self.entry_prices = {ticker: deque() for ticker in tickers}
```

### 2. Récompenser la variation du portfolio
```python
# ❌ MAUVAIS
reward = (new_portfolio - prev_portfolio) / prev_portfolio

# ✅ BON
reward = (prix_vente - prix_achat) / prix_achat
```

### 3. Ne pas forcer la vente à la fin
```python
# ❌ MAUVAIS
if terminated:
    return obs, reward, terminated, truncated, info

# ✅ BON
if terminated:
    # Vendre toutes les positions et ajouter PnL final au reward
    for ticker in self.tickers:
        if self.shares[ticker] > 0:
            # Calculer et ajouter PnL final
            ...
```

---

## 📊 RÉSULTATS ATTENDUS

Après application de ces fixes:

```
🎯 RÉSULTATS UNIVERSALTRADINGENV

💰 PORTFOLIO:
   Moyen : $115,000 (+15%)
   Std   : $5,000

🎯 ACTIONS (tous tickers confondus):
   HOLD  : 50-60%
   BUY   : 20-30%
   SELL  : 15-20%

📈 SHARPE: > 1.0
```

---

## 🚀 PROCHAINES ÉTAPES

1. Implémenter les modifications
2. Tester sur 2-3 tickers d'abord
3. Valider les résultats
4. Étendre à 10+ tickers
5. Ajouter indicateurs techniques progressivement
6. Entraîner modèle production (1M+ steps)

---

**Basé sur**: `DiscreteTradingEnv` (validé le 6 déc 2025)  
**Statut**: ☐ À implémenter
