# 🚀 Ploutos V6 Extended : 50M TIMESTEPS

## 🎯 Pourquoi 50M Steps ?

### 📉 Analyse Performance V6 Standard (15M)

**Résultats actuels :**
```
BUY Timing:  23% good (vs 15% V4)  → +50% amélioration MAIS insuffisant
SELL Timing: 85% good               → Excellent
Return:      +15.87%                → Bon
Objectif:    50%+ BUY good          → Non atteint
```

### 🧠 Hypothèse : Underfitting

**Complexité du problème :**
```
Tickers:       15
Features:      85 par ticker
Dimensions:    1,275 (15 × 85)
Paramètres:    ~2.5M (réseau)

Formule empirique: Timesteps optimal ≈ Tickers × Features × 5000
15 × 85 × 5000 = 6.4M minimum

Règle sûr: 5-8x le minimum = 30M-50M
```

**Conclusion :** 15M est probablement **insuffisant** pour convergence complète.

---

## ⚙️ Configuration V6 Extended

### Modifications vs V6 Standard

| Paramètre | V6 Standard | V6 Extended 50M | Raison |
|-----------|-------------|-----------------|--------|
| **total_timesteps** | 15M | **50M** | Convergence complète |
| **ent_coef** | 0.10 | **0.08** | Moins exploration, plus exploitation |
| **checkpoint_freq** | 50k | **100k** | Moins d'I/O disque |
| **eval_freq** | 10k | **20k** | Moins d'évals, plus de train |
| **early_stopping** | Non | **Oui (25 evals)** | Protection overfit |

### 🛡️ Protection Anti-Overfitting

**1. Early Stopping**
```yaml
early_stopping:
  enabled: true
  max_no_improvement_evals: 25  # Stop si stagnation 500k steps
  min_evals: 100                # Attendre 2M steps minimum
```

**2. Régularisation naturelle**
- `ent_coef: 0.08` - Entropy force exploration
- `max_grad_norm: 0.5` - Clip gradients
- `target_kl: 0.02` - Limite divergence policy

**3. Monitoring continu**
- Évaluation tous les 20k steps
- Best model sauvegardé automatiquement
- Checkpoints tous les 100k steps

---

## 🚀 Lancer l'Entraînement

### Prérequis

**Hardware recommandé :**
- GPU : RTX 3080 / RTX 4070 ou supérieur
- RAM : 32GB+
- Disque : 10GB libres

**Durée estimée :**

| GPU | Durée 50M steps |
|-----|-------------------|
| **RTX 3080** | **15-18h** ⏱️ |
| **RTX 4090** | 8-10h ⚡ |
| **CPU 16 cores** | 120-150h 🐢 |

### Commandes

```bash
cd /root/ai-factory/tmp/project_ploutos

# Récupérer derniers fichiers
git pull origin main

# Rendre exécutable
chmod +x scripts/train_v6_extended_50m.sh

# ✅ RECOMMANDÉ: Mode background
bash scripts/train_v6_extended_50m.sh --nohup

# Suivre progression
tail -f logs/v6_extended_50m/training_*.log

# Monitorer GPU
watch -n 1 nvidia-smi

# TensorBoard
tensorboard --logdir runs/v6_extended_50m/ --port 6006
```

### Arrêter l'Entraînement

```bash
# Trouver le processus
ps aux | grep train_v6_extended_50m

# Arrêter proprement (sauvegarde dernier checkpoint)
pkill -f train_v6_extended_50m

# Ou kill brutal (pas recommandé)
kill -9 <PID>
```

---

## 📊 Monitoring

### Logs à Surveiller

**1. Training logs**
```bash
tail -f logs/v6_extended_50m/training_*.log

# Chercher:
# - "rollout/ep_rew_mean" : reward moyen
# - "train/entropy_loss" : exploration
# - "train/policy_loss" : convergence
```

**2. TensorBoard**
```bash
tensorboard --logdir runs/v6_extended_50m/

# Ouvrir: http://localhost:6006
# Surveiller:
# - rollout/ep_rew_mean : doit monter progressivement
# - train/entropy_loss : doit diminuer légèrement
# - eval/mean_reward : NE DOIT PAS diverger de train
```

**3. GPU Usage**
```bash
watch -n 1 nvidia-smi

# Vérifier:
# - GPU Util : 90-100% (bon)
# - Memory : 8-12GB utilisés (normal)
# - Temperature : <85°C (ok)
```

### 🚨 Signaux d'Alerte

**⚠️ Overfitting détecté :**
```
train/ep_rew_mean : 150 (monte)
eval/mean_reward  : 80  (descend ou stagne)
→ GAP se creuse = OVERFITTING
→ Early stopping va s'activer
```

**⚠️ Instabilité :**
```
train/policy_loss : valeurs erratiques
train/value_loss  : explose
→ Learning rate trop élevé
→ Considérer réduire LR
```

**✅ Tout va bien :**
```
train/ep_rew_mean : monte doucement
eval/mean_reward  : suit train (gap <20%)
train/entropy_loss : diminue légèrement
→ Convergence saine
```

---

## 🎯 Objectifs V6 Extended 50M

### Métriques Cibles

| Métrique | V4 | V6 15M | **V6 50M Cible** |
|----------|-------|--------|------------------|
| **BUY Quality** | 15% | 23% | **45-60%** 🎯 |
| **SELL Quality** | 60% | 85% | **80-90%** ✅ |
| **Return** | +7.4% | +15.9% | **+20%+** 🚀 |
| **Sharpe Ratio** | 1.59 | 4.27 | **>4.0** ✅ |
| **Win Rate** | 50% | 43% | **55-65%** 🎯 |
| **Max Drawdown** | 4.5% | 5.6% | **<7%** ✅ |

### Critères de Succès

✅ **Succès COMPLET** : BUY quality ≥ 50%  
✅ **Succès PARTIEL** : BUY quality ≥ 35%  
❌ **Échec** : BUY quality < 30%  

---

## 🧪 Tester le Modèle

### Après Entraînement

**1. Backtest de performance**
```bash
python scripts/backtest_v6.py \
    --model models/v6_extended_50m_best/best_model.zip \
    --episodes 10 \
    --days 90
```

**2. Analyse timing (CRITIQUE)**
```bash
python scripts/analyze_why_fails_v6.py \
    --model models/v6_extended_50m_best/best_model.zip
```

**Chercher dans les résultats :**
```
🎯 Analyse qualité du timing...

  📈 BUYs:
    ✅ Good (buy low):  ??? (??.?%)   ← DOIT ÊTRE ≥ 45%
    ❌ Bad (buy high):  ??? (??.?%)
```

**3. Comparaison vs V6 15M**
```bash
# Tester les 2 modèles sur mêmes données
python scripts/compare_models.py \
    --model1 models/v6_better_timing_best/best_model.zip \
    --model2 models/v6_extended_50m_best/best_model.zip
```

---

## 🔄 Si Échec (BUY Quality < 30%)

### Plan B : Approches Alternatives

**1. Simplifier drastiquement (V7 Ultra-Simple)**
- 1 ticker (NVDA uniquement)
- 10-15 features max
- Reward = PnL pur
- 10M timesteps

**2. Reward Shaping plus agressif**
- Bonus explicite +5.0 pour good BUY timing
- Pénalité -2.0 pour bad BUY timing
- Utiliser lookahead pendant training

**3. Approche Hybride RL + Rules**
- RL décide **QUAND** trader (timing)
- Règles fixes **COMBIEN** trader (sizing)
- Combine meilleur des 2 mondes

**4. Algorithmes alternatifs**
- DQN (Discrete actions, plus simple)
- SAC (Soft Actor-Critic, plus stable)
- A2C (Advantage Actor-Critic)

---

## 📚 Références

### Fichiers Clés

**Configuration :**
- `config/training_config_v6_extended_50m.yaml`

**Scripts :**
- `training/train_v6_extended_50m.py`
- `scripts/train_v6_extended_50m.sh`
- `scripts/backtest_v6.py`
- `scripts/analyze_why_fails_v6.py`

**Environnement :**
- `core/universal_environment_v6_better_timing.py`
- `core/advanced_features_v2.py`

### Documentation Complémentaire

- [Guide V6 Standard](./V6_BETTER_TIMING.md)
- [Features V2 Details](../core/advanced_features_v2.py)
- [Training Best Practices](./TRAINING.md)

---

## ⏱️ Timeline Estimée

**RTX 3080 (cas typique) :**

```
Hébergement: 0h    ─── Début entraînement
H+3h:               ─── ~10M steps (20%)
H+6h:               ─── ~20M steps (40%)
H+9h:               ─── ~30M steps (60%)
H+12h:              ─── ~40M steps (80%)
H+15h:              ─── ~50M steps (100%) ✅

Ou early stopping si convergence avant!
```

---

**Date :** December 11, 2025  
**Version :** V6 Extended 50M  
**Status :** ✅ Prêt à lancer  
**Objectif :** BUY quality ≥ 45%
