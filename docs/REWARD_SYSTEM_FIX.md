# 🎯 FIX DU SYSTÈME DE REWARD - DOCUMENTATION COMPLÈTE

## 📊 RÉSUMÉ EXÉCUTIF

**Date**: 6 décembre 2025
**Problème**: L'IA ne tradait pas (portfolio = $100k constant, 100% actions SELL)
**Solution**: Reward basé sur PnL réalisé + Actions discrètes
**Résultat**: +14% profit, actions équilibrées (28% BUY / 15% SELL)

---

## 🔴 LE PROBLÈME

### **Symptom init ial**
```
Portfolio: $100,000 (+0.0%)
Actions: BUY 0% / SELL 100% / HOLD 0%
Sharpe: 0.000
```

L'IA ne faisait **AUCUN trade utile**. Elle spammait l'action SELL même sans positions.

### **Root Cause #1: Reward Function Incorrecte**

**Ancien système** (`universal_environment.py`) :
```python
reward = (new_portfolio - previous_portfolio) / previous_portfolio
```

**Problème** :
- Chaque trade coûte ~0.01% en commission
- Le reward est TOUJOURS négatif (même si le prix monte)
- L'IA apprend que "ne rien faire = meilleur choix"

**Exemple concret** :
```python
# Trade BUY:
Step 1: BUY 185 shares @ $538
  - Portfolio AVANT: $100,000
  - Portfolio APRÈS: $99,990 (frais: -$10)
  - Reward: -0.0001  ❌ NÉGATIF

# Trade SELL:
Step 2: SELL 185 shares @ $536 
  - Portfolio AVANT: $99,990
  - Portfolio APRÈS: $99,588 (frais: -$10, perte prix: -$392)
  - Reward: -0.0001  ❌ NÉGATIF
```

**Conclusion de l'IA** : *"Trading = reward négatif. Meilleure stratégie = ne rien faire."*

### **Root Cause #2: Action Space Ambigu**

**Ancien système** :
```python
action_space = Box(low=-1, high=1, shape=(n_assets,))
```

Qu'est-ce que `action=0.7` veut dire ? L'IA ne le sait pas clairement.

**Résultat** :
- L'IA converge vers `action=-1` (SELL maximal) comme action "par défaut"
- Elle n'explore jamais les autres valeurs

---

## ✅ LA SOLUTION

### **1. Reward = PnL RÉALISÉ (pas portfolio total)**

**Nouveau système** (`discrete_trading_env.py`) :
```python
if action == 1:  # BUY
    # Enregistrer prix d'entrée
    for _ in range(shares_bought):
        self.entry_prices.append(current_price)
    reward = 0  # ✅ On attend le résultat

elif action == 2:  # SELL
    # Calculer PnL réalisé
    pnl_total = 0
    for _ in range(shares_sold):
        entry_price = self.entry_prices.popleft()
        pnl = (current_price - entry_price) / entry_price
        pnl_total += pnl
    
    avg_pnl = pnl_total / shares_sold
    reward = avg_pnl  # ✅ Positif si profit, négatif si perte

else:  # HOLD
    reward = 0  # Neutre
```

**Avantages** :
- Signal **CLAIR** : Acheter bas + vendre haut = reward positif
- Les frais sont un coût réel mais n'impactent pas le reward
- L'IA comprend la **QUALITÉ** de ses décisions

### **2. Reward sur PnL LATENT (encourage holding)**

```python
if action == 0 and self.shares > 0:  # HOLD avec position
    avg_entry = np.mean(list(self.entry_prices))
    unrealized_pnl = (current_price - avg_entry) / avg_entry
    reward = unrealized_pnl * 0.005  # ✅ Petit bonus (0.5%)
```

**Effet** : L'IA est récompensée pour **tenir** des positions gagnantes.

### **3. Actions Discrètes (HOLD/BUY/SELL)**

```python
action_space = Discrete(3)
# 0 = HOLD  (ne rien faire)
# 1 = BUY   (acheter 20% du portfolio)
# 2 = SELL  (vendre TOUT)
```

**Avantages** :
- Signal **ultra-clair** pour l'IA
- Pas d'ambiguïté sur l'intention
- Force l'exploration de SELL

### **4. Vente Forcée à la Fin**

```python
if (terminated or truncated) and self.shares > 0:
    # Forcer la clôture de toutes les positions
    avg_entry = np.mean(list(self.entry_prices))
    final_pnl = (current_price - avg_entry) / avg_entry
    reward += final_pnl  # ✅ PnL final
```

**Effet** : L'IA DOIT évaluer ses positions (pas de "buy & hold forever").

---

## 📈 RÉSULTATS

### **Avant (système cassé)** :
```
Portfolio: $100,000 (+0.0%)
Actions:
  BUY   : 0.0%
  SELL  : 100.0%
  HOLD  : 0.0%
Sharpe: 0.000

Diagnostic: L'IA ne trade pas
```

### **Après (système fixé)** :
```
Portfolio: $113,723 (+13.7%)
Variance: $3,883 (Min $98k / Max $123k)
Actions:
  BUY   : 28.3%
  SELL  : 14.7%
  HOLD  : 57.0%
Sharpe: 10.0

Diagnostic: L'IA trade ET gagne de l'argent ✅
```

---

## 💻 FICHIERS CLÉS

### **Environnement Validé**
- **`core/discrete_trading_env.py`** - Environnement avec actions discrètes + reward PnL

### **Script de Test Validé**
- **`scripts/test_discrete_env.py`** - Test complet avec SPY (500 jours)

### **Fichiers Obsolètes (\u00e0 nettoyer)** :
- `core/simple_pnl_environment.py` - Version intermédiaire (actions continues)
- `scripts/debug_simple_env.py` - Debug environnement minimal
- `scripts/debug_verbose_env.py` - Debug step-by-step
- `scripts/test_pnl_reward.py` - Test reward PnL (continuous)

---

## 🚀 APPLICATION À UNIVERSALTRADINGENV

### **Étape 1 : Adapter Action Space**

**Actuel** :
```python
action_space = Box(low=-1, high=1, shape=(n_assets,))
```

**Nouveau** :
```python
from gymnasium.spaces import MultiDiscrete

# 3 actions (HOLD/BUY/SELL) par asset
action_space = MultiDiscrete([3] * n_assets)
```

### **Étape 2 : Implémenter Reward PnL par Ticker**

```python
class UniversalTradingEnv(gym.Env):
    def __init__(self, ...):
        # Tracking PnL par ticker
        self.entry_prices = {ticker: deque() for ticker in tickers}
        self.entry_steps = {ticker: None for ticker in tickers}
    
    def step(self, action):
        total_reward = 0
        
        for i, ticker in enumerate(self.tickers):
            action_val = action[i]  # 0=HOLD, 1=BUY, 2=SELL
            
            if action_val == 1:  # BUY
                # Enregistrer prix d'entrée
                self.entry_prices[ticker].append(current_price)
                # Reward = 0
            
            elif action_val == 2:  # SELL
                # Calculer PnL
                pnl = (current_price - avg_entry) / avg_entry
                total_reward += pnl / self.n_assets
            
            else:  # HOLD
                # Reward sur PnL latent
                if self.positions[ticker] > 0:
                    unrealized_pnl = ...
                    total_reward += unrealized_pnl * 0.005 / self.n_assets
        
        return obs, total_reward, terminated, truncated, info
```

### **Étape 3 : Tester**

```bash
# Adapter train.py pour utiliser le nouvel environnement
python3 train.py --env discrete --tickers SPY QQQ --timesteps 500000
```

---

## 📝 LESSONS LEARNED

### **1. Le Reward doit refléter la QUALITÉ de la décision, pas le résultat brut**

❌ **Mauvais** : Reward = variation portfolio (inclut frais, timing, chance)
✅ **Bon** : Reward = PnL du trade (performance pure de la décision)

### **2. Les frais de transaction dominent le signal si mal gérés**

Si commission = 0.01% et reward = variation portfolio :
- Chaque trade = -0.01% minimum
- Le signal de prix (\u00b10.1%) est **noyé** par le bruit des frais

### **3. Actions discrètes > Actions continues pour le trading**

Le trading est une décision **binaire** : acheter, vendre, ou ne rien faire.
Un action space continue ajoute de la complexité inutile.

### **4. Récompenser le HOLDING de positions gagnantes**

Sans reward sur PnL latent, l'IA peut :
- Acheter et **ne jamais vendre** (buy & hold passif)
- Ou vendre trop tôt (peur du risque)

Le reward latent (0.5% du PnL) encourage la **patience**.

---

## ⚠️ ATTENTION

### **Ne PAS modifier `universal_environment.py` directement**

Ce fichier est utilisé en production. Toute modification peut casser le système existant.

**Plan recommandé** :
1. Créer `universal_discrete_environment.py` (nouveau fichier)
2. Tester en parallèle avec `discrete_trading_env.py`
3. Valider sur multi-assets
4. Migrer progressivement

---

## 🏁 NEXT STEPS

### **Court terme** :
- [ ] Nettoyer fichiers debug temporaires
- [ ] Créer `universal_discrete_environment.py`
- [ ] Tester avec 3-5 assets simultanément

### **Moyen terme** :
- [ ] Entraîner sur 1M steps (au lieu de 200k)
- [ ] Sauvegarder modèle validé (`models/discrete_v1.zip`)
- [ ] Backtester sur données hors-sample

### **Long terme** :
- [ ] Migrer vers Multi-Asset Discrete Env
- [ ] Ajouter indicateurs (RSI, MACD) à l'observation
- [ ] Implémenter reward shaping avancé (Sharpe, drawdown)

---

## 📚 RÉFÉRENCES

- **OpenAI Spinning Up** - [Reward Design](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#reward-and-return)
- **Stable-Baselines3** - [Custom Environments](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html)
- **Gymnasium** - [Discrete Action Space](https://gymnasium.farama.org/api/spaces/fundamental/#discrete)

---

**Auteur**: Thomas BOISAUBERT  
**Date**: 6 décembre 2025  
**Projet**: Ploutos Trading IA  
**Statut**: ✅ VALIDÉ
