# 🔧 V3 FIXED - CORRECTIONS BUGS CRITIQUES

## 🚨 BUGS TROUVÉS DANS V3_ULTIMATE

### **1. Overtrading Massif (290,640 trades sur 90j)** ❌

**Problème** :
```python
# Ligne 215 de universal_environment_v3_trend.py
current_day = self.current_step // 6  # ❌ FAUX !
```

- **Cause** : Code assum ait données HORAIRES (6 steps/jour)
- **Réalité** : Données sont DAILY (1 step = 1 jour)
- **Résultat** : `trades_today` ne se reset JAMAIS correctement
- **Impact** : 3,229 trades/jour au lieu de max 50 !

**Solution V3 FIXED** :
```python
# Ligne 166 de universal_environment_v3_fixed.py
if self.current_step != self.current_date_step:
    self.trades_today = 0
    self.current_date_step = self.current_step
```

---

### **2. Lookahead Bias = TRICHE** ❌

**Problème** :
```python
# Lignes 478-485 de universal_environment_v3_trend.py
if idx + self.lookahead_steps < len(self.precomputed[ticker]['close']):
    future_price = self.precomputed[ticker]['close'][idx + self.lookahead_steps]
    future_return = (future_price - current_price) / current_price
    
    if future_return > 0.01:
        reward += 0.1  # BONUS anticipation ❌ TRICHE !
```

- **Cause** : Modèle voit le futur pendant entraînement
- **Résultat** : Overfit massif, modèle inutilisable en prod
- **Impact** : Score backtest -13.73% (catastrophique)

**Solution V3 FIXED** :
```python
# Lignes 558-562 de universal_environment_v3_fixed.py
# ✅ Reward basé PnL réel (NO LOOKAHEAD)
reward = avg_pnl * 2.0  # Multiplier pour importance

# BONUS: Vendre avant baisse (indicateurs techniques SEULEMENT)
if trend_daily < 0 or trend_weekly < 0:
    reward += 0.05  # Bonus sortie avant tendance négative
```

---

### **3. Reward Clipping Trop Strict** ❌

**Problème** :
```python
# Ligne 266 de universal_environment_v3_trend.py
total_reward = np.clip(total_reward, -0.5, 0.5)  # ❌ Trop strict !
```

- **Cause** : Clipping écrase grosses pertes/gains
- **Résultat** : Modèle ne comprend pas gravité erreurs
- **Impact** : Apprentissage médiocre

**Solution V3 FIXED** :
```python
# Ligne 254 de universal_environment_v3_fixed.py
total_reward = np.clip(total_reward, -2.0, 2.0)  # ✅ Range large !

# Ligne 580 (par action)
return np.clip(reward, -0.5, 0.5)  # Clip par action, pas total
```

---

## ✅ AMÉLIORATIONS V3 FIXED

### **1. Observation Space Enrichi : 115 features** 🎯

**V3_ULTIMATE** : 107 features  
**V3_FIXED** : 115 features

**Nouvelles features** :
```python
# 13 features/ticker (au lieu de 11)
- Bollinger Bands position
- MACD diff
- Stochastic normalized

# 5 features portfolio (au lieu de 3)
- Cash ratio
- Total value norm
- N positions
- Drawdown       # ✨ NEW
- Sharpe approx  # ✨ NEW
```

**Avantage** : Modèle voit mieux risque et performance

---

### **2. Position Sizing Optimisé** 📊

```python
# Lignes 496-502 de universal_environment_v3_fixed.py
if self.use_smart_sizing:
    volatility_factor = 1.0 / (1.0 + atr * 4.0)
    confidence_factor = max(0.3, min((adx + 1.0) / 2.5, 1.0))
    position_pct = self.buy_pct * volatility_factor * confidence_factor
    position_pct = np.clip(position_pct, 0.03, 0.25)
```

**Améliorations** :
- Position plus petite si volatilité élevée (ATR)
- Position ajustée selon confiance (ADX)
- Range 3% à 25% (au lieu de fixe 20%)

---

### **3. Stop-Loss / Take-Profit Adaptatifs** 🛑

```python
# Lignes 189-193 de universal_environment_v3_fixed.py
atr = self.precomputed[ticker]['atr_norm'][self.current_step]
stop_loss_adjusted = self.stop_loss_pct * (1.0 + atr * 2.0)  # -3% à -10%
take_profit_adjusted = self.take_profit_pct * (1.0 + atr)     # +15% à +30%
```

**Logique** :
- Actions volatiles : Stop-loss plus large, Take-profit plus haut
- Actions stables : Stop-loss serré, Take-profit proche

---

### **4. Rewards Intelligents** 🎯

**Actions pénalisées** :
```python
# Lignes 511-523
if trend_daily < 0:  # Contre tendance daily
    reward -= 0.08
if trend_weekly < 0:  # Contre tendance weekly
    reward -= 0.05
if spy_trend < 0:  # Marché baissier
    reward -= 0.05
if vix_level > 1.0:  # VIX > 30 (panique)
    reward -= 0.04
if rsi > 0.6:  # RSI > 80 (surachat)
    reward -= 0.03
if bb_pos > 0.9:  # Prix haut Bollinger
    reward -= 0.03
```

**Actions bonifiées** :
```python
# Lignes 525-529
if trend_daily > 0 and trend_weekly > 0:
    reward += 0.05
if rsi < -0.4 and bb_pos < 0.3:  # Survente + bas Bollinger
    reward += 0.04
```

---

## 🚀 UTILISATION

### **Entraînement V3 FIXED**

```bash
# Sur machine BBC (GPU)
cd /root/ai-factory/tmp/project_ploutos
source /root/ai-factory/venv/bin/activate

# Entraînement standard 10M steps
python3 scripts/train_v3_fixed.py --wandb --project Ploutos_V3_FIXED

# Logs
tail -f logs/train_v3_fixed.log
```

**Paramètres** :
- `--steps` : Timesteps total (défaut 10M)
- `--envs` : Environnements parallèles (défaut 64)
- `--max-trades-per-day` : Limite trades (défaut 30)
- `--wandb` : Activer W&B tracking
- `--project` : Nom projet W&B

**Durée estimée** :  
- 10M steps avec 64 envs = ~12-15h sur RTX 3080

---

### **Backtest**

```bash
# Backtest 90 jours
python3 scripts/backtest_reliability.py \
  --model models/ploutos_v3_fixed.zip \
  --days 90 \
  --episodes 5

# Backtest 365 jours
python3 scripts/backtest_reliability.py \
  --model models/ploutos_v3_fixed.zip \
  --days 365 \
  --episodes 10
```

---

## 🎯 OBJECTIFS V3 FIXED

| Métrique | V2 | V3_ULTIMATE | **V3_FIXED Cible** |
|----------|----|--------------|-----------------|
| **Score 90j** | 91.8 | 45.4 ❌ | **>90** ✅ |
| **Score 365j** | 45.3 | Pas testé | **>80** ✅ |
| **Return 90j** | +66% | -13.7% ❌ | **>50%** ✅ |
| **Return 365j** | -28% | Pas testé | **>20%** ✅ |
| **Drawdown** | 31% | 23.8% | **<8%** ✅ |
| **Trades/jour** | 640 ❌ | 3,229 ❌❌ | **<30** ✅ |
| **Win Rate** | 52% | 53.1% | **>55%** ✅ |

---

## 📊 RÉSULTATS ATTENDUS

### **Correction Overtrading**

```
V3_ULTIMATE: 290,640 trades / 90j = 3,229 trades/jour ❌
V3_FIXED:    2,700 trades / 90j = 30 trades/jour ✅
```

**Impact** :  
- Commissions réduites de 97%  
- Trades plus réfléchis
- Performance améliorée

### **Correction Lookahead**

```
V3_ULTIMATE: Voit futur pendant entraînement ❌
             → Overfit massif
             → Échec en production

V3_FIXED:    Entraînement honnête ✅
             → Généralise bien
             → Performance stable
```

### **Rewards Optimisés**

```
V3_ULTIMATE: Range [-0.5, +0.5] ❌
             → Grosses erreurs écrasées
             → Apprentissage médiocre

V3_FIXED:    Range [-2.0, +2.0] ✅
             → Erreurs graves pénalisées fortement
             → Apprentissage efficace
```

---

## 🛠️ COMPARAISON CODE

### **Trades per day**

| Version | Code |
|---------|------|
| **V3_ULTIMATE** | `current_day = self.current_step // 6` ❌ |
| **V3_FIXED** | `if self.current_step != self.current_date_step:` ✅ |

### **Rewards**

| Version | BUY Lookahead | SELL Lookahead | Range |
|---------|---------------|----------------|-------|
| **V3_ULTIMATE** | `if future_return > 0.01: reward += 0.1` ❌ | `if future_return < -0.01: reward += 0.1` ❌ | [-0.5, +0.5] |
| **V3_FIXED** | AUCUN ✅ | AUCUN ✅ | [-2.0, +2.0] |

### **Observation**

| Version | Features | Détail |
|---------|----------|--------|
| **V3_ULTIMATE** | 107 | 11/ticker + 2 marché + 3 portfolio |
| **V3_FIXED** | 115 | 13/ticker + 2 marché + 5 portfolio ✅ |

---

## ✅ CHECKLIST POST-ENTRAÎNEMENT

- [ ] Modèle sauvegardé : `models/ploutos_v3_fixed.zip`
- [ ] Config sauvegardée : `models/ploutos_v3_fixed.json`
- [ ] Checkpoints présents : `models/production_v3_fixed/checkpoints/`
- [ ] Backtest 90j : Score >90
- [ ] Backtest 365j : Score >80
- [ ] Trades/jour : <30
- [ ] Drawdown : <8%
- [ ] Win rate : >55%
- [ ] Return 365j : >20%

---

## 📝 NOTES

### **Pourquoi V3_ULTIMATE a échoué ?**

1. **Overtrading** : Bug compteur trades (division par 6)
2. **Lookahead bias** : Modèle trichait pendant entraînement  
3. **Rewards faibles** : Erreurs graves pas assez pénalisées

### **Différences V3_FIXED**

1. ✅ Compteur trades CORRIGÉ (DAILY data)
2. ✅ NO lookahead (entraînement honnête)
3. ✅ Rewards larges (apprentissage efficace)
4. ✅ Observation enrichie (115 features)
5. ✅ Position sizing optimisé (ATR + ADX)
6. ✅ Stop-loss/Take-profit adaptatifs

---

## 🚀 PROCHAINES ÉTAPES

1. **Lancer entraînement V3 FIXED** (12-15h)
2. **Backtest 90j** (vérifier score >90)
3. **Backtest 365j** (vérifier score >80)
4. **Si OK** : Déployer sur VPS en paper trading
5. **Monitorer 7 jours** (vérifier comportement)
6. **Si stable** : Passage LIVE

---

**Date** : 9 Décembre 2025  
**Auteur** : Ploutos AI Team  
**Version** : V3 FIXED  
**Status** : ✅ PRÊT POUR ENTRAÎNEMENT
