# 🎯 Ploutos V6 : BETTER TIMING

## 🔍 Diagnostic du Problème

### Analyse Approfondie (Dec 2025)

Après analyse du modèle V4, on a identifié **LE** problème :

```
📊 Distribution globale:
  • HOLD (0): 4,005 (89.0%)  ✅ OK
  • BUY  (1): 261 (5.8%)    ✅ OK  
  • SELL (2): 234 (5.2%)    ✅ OK

📊 BUYs:
  ✅ Good (buy low):  355 (15.4%)   ❌ PROBLÈME
  ❌ Bad (buy high):  1,950 (84.6%)

📊 SELLs:
  ✅ Good (sell high): 1,335 (59.9%)  ✅ BON
  ❌ Bad (sell low):  892 (40.1%)
```

**Conclusion :** L'IA **trade** activement (4,532 actions), **sait vendre** au bon moment (60% good), mais **achète trop tard** (85% buy high).

### ❌ Causes Identifiées

1. **Features Momentum Inefficaces**
   - RSI, MACD donnent signaux **après** le mouvement
   - L'IA voit "prix monte" → achète → trop tard

2. **Pas de Détection de Reversal**
   - Manque features pour détecter **début** de mouvement
   - Pas de détection support/resistance

3. **Lookback Period Trop Court**
   - Features calculées sur trop peu de données
   - Manque contexte macro

---

## ✅ Solution : Features V2

### Nouvelles Features (60+ par ticker)

#### 1. Support/Resistance Dynamiques
```python
# 3 timeframes: 20, 50, 100 barres
- support_20, support_50, support_100
- resistance_20, resistance_50, resistance_100
- dist_support_* : distance actuelle vs support
- dist_resistance_* : distance actuelle vs resistance
- near_support_* : signal BUY si proche support (<2%)
```

#### 2. Mean Reversion Signals
```python
# Détecte quand prix s'éloigne trop de la moyenne
- zscore_20, zscore_50 : distance en écart-types
- oversold_20, oversold_50 : z-score < -1.5 = BUY
- overbought_20, overbought_50 : z-score > 1.5 = SELL
- reverting_20, reverting_50 : prix commence à revenir
```

#### 3. Volume Confirmation
```python
# Volume confirme la force du mouvement
- vol_ratio : volume actuel vs moyenne
- vol_spike : volume > 1.5x moyenne
- vol_bullish : volume + prix monte = confirmation
- vol_bearish : volume + prix baisse = confirmation
- vol_low : volume < 0.7x moyenne = manque conviction
```

#### 4. Price Action Patterns
```python
# Patterns de chandeliers pour reversal
- hammer : bullish reversal (long lower wick)
- shooting_star : bearish reversal (long upper wick)
- doji : indecision
- bullish_engulfing : pattern bullish fort
- bearish_engulfing : pattern bearish fort
```

#### 5. Divergences RSI/Prix
```python
# Divergence = signal fort de reversal
- bullish_divergence : prix fait lower low, RSI fait higher low
- bearish_divergence : prix fait higher high, RSI fait lower high
```

#### 6. Bollinger Patterns
```python
# Squeeze, breakout, etc.
- bb_position : position dans les bandes (0-1)
- touch_lower_bb : signal BUY si touche bande basse
- touch_upper_bb : signal SELL si touche bande haute
- bb_squeeze : bandes se resserrent = breakout imminent
```

#### 7. 🎯 Entry Score Composite
```python
# SCORE D'ENTRÉE qui combine tous les signaux

buy_score = sum([
    near_support, oversold, reverting,
    vol_bullish, hammer, bullish_engulfing,
    bullish_divergence, touch_lower_bb
])

sell_score = sum([
    near_resistance, overbought,
    shooting_star, bearish_engulfing,
    bearish_divergence, touch_upper_bb
])

entry_signal = buy_score_norm - sell_score_norm
```

#### 8. Momentum Amélioré
```python
# Détecte DÉBUT de momentum (pas fin)
- momentum_accel_* : accélération du momentum
- momentum_start_* : début momentum (accel+ & momentum faible)
```

#### 9. Trend Strength (ADX)
```python
# Force du trend
- adx : Average Directional Index
- strong_trend : ADX > 25
- weak_trend : ADX < 20
```

#### 10. Régime de Volatilité
```python
# Adapter stratégie selon volatilité
- atr_pct : ATR en % du prix
- high_vol : haute volatilité (top 30%)
- low_vol : basse volatilité (bottom 30%)
```

---

## 🚀 Entraînement V6

### Configuration

**Environnement : V6 BetterTiming**
- 60+ features par ticker (vs 37 avant)
- Entry score composite
- Support/Resistance dynamiques

**Training Config :**
```yaml
training:
  total_timesteps: 15000000  # 15M
  n_envs: 16
  batch_size: 8192
  n_epochs: 20
  learning_rate: 0.0001
  ent_coef: 0.10  # Exploration modérée

environment:
  buy_pct: 0.20
  max_position_pct: 0.25
  max_trades_per_day: 10
  min_holding_period: 2
  reward_scaling: 1.5
```

### Lancer l'Entraînement

```bash
cd /root/ai-factory/tmp/project_ploutos

# Récupérer derniers fichiers
git pull origin main

# Rendre exécutable
chmod +x scripts/train_v6_better_timing.sh

# Option 1 : Mode interactif
bash scripts/train_v6_better_timing.sh

# Option 2 : Mode background (recommandé)
bash scripts/train_v6_better_timing.sh --nohup

# Suivre logs (si background)
tail -f logs/v6_better_timing/training_*.log

# Monitorer GPU
watch -n 1 nvidia-smi

# TensorBoard
tensorboard --logdir runs/v6_better_timing/ --port 6006
```

### Durée d'Entraînement

| Hardware | Durée 15M steps |
|----------|------------------|
| **RTX 3080** | 5-7h ⚡ |
| **RTX 4090** | 3-4h 🚀 |
| **CPU 16 cores** | 36-48h 🐢 |

---

## 🎯 Résultats Attendus

### Améliorations Cibles

**BUY Timing (CRITIQUE) :**
```
Actuel:  15% good buys
Cible:   60%+ good buys  🎯
```

**SELL Timing (DÉJÀ BON) :**
```
Actuel:  60% good sells
Cible:   65%+ good sells  ✅
```

**Performance Globale :**
```
Actuel:  +7.4% vs Buy&Hold +5.6% (+1.8% outperform)
Cible:   +15%+ vs Buy&Hold +5.6% (+10%+ outperform)  🚀
```

### Indicateurs de Succès

✅ **BUY quality > 50%** (vs 15% actuel)  
✅ **Win rate > 60%** (vs ~50% actuel)  
✅ **Sharpe ratio > 2.0** (vs 1.59 actuel)  
✅ **Max drawdown < 10%** (vs 4.5% actuel, OK)  
✅ **Outperformance > +5%** (vs +1.8% actuel)  

---

## 🧪 Tester le Modèle

### Backtest de Fiabilité

```bash
# Après entraînement, tester le modèle
python scripts/backtest_reliability.py \
    --model models/v6_better_timing_best/best_model.zip \
    --episodes 5 \
    --days 90
```

### Analyse Approfondie

```bash
# Analyser timing des trades
python scripts/analyze_why_fails.py \
    --model models/v6_better_timing_best/best_model.zip
```

**Vérifier :**
```
📊 BUYs:
  ✅ Good (buy low):  ??? (??.?%)   ← DOIT ÊTRE > 50%
  ❌ Bad (buy high):  ??? (??.?%)
```

---

## 📈 Comparaison Versions

| Version | Features/Ticker | BUY Quality | SELL Quality | Outperform |
|---------|----------------|-------------|--------------|------------|
| **V3** | 10 | ? | ? | ? |
| **V4 Ultimate** | 37 | **15%** ❌ | **60%** ✅ | **+1.8%** |
| **V6 Better Timing** | **60+** | **50%+** 🎯 | **65%+** 🎯 | **+10%+** 🎯 |

---

## 💡 Prochaines Étapes

### Si V6 Réussit (BUY quality > 50%)

1. **Déployer en Production**
   - Migrer sur VPS
   - Live trading (paper d'abord)

2. **Optimisations Supplémentaires**
   - Fine-tuning hyperparams
   - Augmenter capital par trade
   - Tester sur plus de tickers

### Si V6 Échoue Encore (BUY quality < 40%)

1. **Simplifier Drastiquement**
   - 1 seul ticker (NVDA)
   - 3 actions simples
   - Reward = PnL uniquement

2. **Approche Hybride**
   - RL décide QUAND trader
   - Règles fixes décident COMBIEN

3. **Exploration Modèles Alternatifs**
   - DQN, A2C, SAC
   - Transformers pour séquences
   - Ensemble methods

---

## 📚 Références

### Fichiers Clés

**Features :**
- `core/advanced_features_v2.py` - 60+ features optimisées

**Environnement :**
- `core/universal_environment_v6_better_timing.py` - Env V6

**Training :**
- `config/training_config_v6_better_timing.yaml` - Config
- `training/train_v6_better_timing.py` - Script Python
- `scripts/train_v6_better_timing.sh` - Lanceur bash

**Analyse :**
- `scripts/analyze_why_fails.py` - Diagnostic approfondi
- `scripts/backtest_reliability.py` - Backtest complet
- `scripts/diagnose_model.py` - Test distribution actions

### Documentation

- [Features V2 Details](../core/advanced_features_v2.py)
- [Training Guide](./TRAINING.md)
- [Backtest Guide](./BACKTEST.md)

---

**Date :** December 10, 2025  
**Version :** V6 Better Timing  
**Status :** 🚧 En développement
