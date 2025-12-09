# 🚀 GUIDE V4 OPTIMAL - ENTRAÎNEMENT PARFAIT

## 🎯 OBJECTIF

Script d'entraînement **PARFAIT** basé sur:
- 📚 Recherche académique PPO (OpenAI Spinning Up)
- 🧠 Expérience V1/V2/V3 (3 mois de tests)
- 📈 Best practices trading bots

---

## ✨ NOUVEAUTÉS V4

### **1. Early Stopping** 🛑

```python
# Stop si pas d'amélioration après 5 evals
patience = 5
min_improvement = 0.02  # 2% minimum
```

**Pourquoi ?**
- Évite overfitting (modèle apprend par cœur)
- Gagne du temps (arrête quand converge)
- Meilleure généralisation

---

### **2. Train/Validation Split** 📏

```python
# 80% train, 20% validation
train_data = data[:80%]
val_data = data[80%:]
```

**Pourquoi ?**
- Évalue sur données NON VUES
- Détecte overfitting précocement
- Garantit généralisation

---

### **3. Learning Rate Scheduler** 📉

```python
# Démarre à 2.5e-4, finit à 2.5e-5
initial_lr = 2.5e-4
final_lr = 2.5e-5  # /10
```

**Pourquoi ?**
- Début : Exploration rapide (LR haut)
- Fin : Convergence fine (LR bas)
- Meilleure performance finale

---

### **4. Best Model Auto-Save** 💾

```python
# Sauvegarde automatique si amélioration
if new_reward > best_reward:
    model.save("best_model.zip")
```

**Avantage** :
- Garde TOUJOURS meilleur modèle
- Pas de perte si crash
- Facile à déployer

---

### **5. Config Académique Optimale** 🎯

Basée sur **"Spinning Up in Deep RL"** (OpenAI) :

```python
OPTIMAL_CONFIG = {
    'learning_rate': 2.5e-4,  # OpenAI optimal
    'batch_size': 64,         # Petit = meilleure généralisation
    'n_epochs': 10,           # Standard PPO
    'net_arch': [512, 512, 256],  # Plus petit = moins overfit
    'ent_coef': 0.005,        # Exploration modérée
}
```

---

## 📊 COMPARAISON V3 vs V4

| Feature | V3 FIXED | **V4 OPTIMAL** |
|---------|----------|----------------|
| **Early stopping** | ❌ | ✅ Oui (patience 5) |
| **Train/Val split** | ❌ | ✅ 80/20 |
| **LR scheduler** | ❌ | ✅ Linear decay |
| **Best model save** | ❌ | ✅ Automatique |
| **Batch size** | 4096 | **64** (meilleur) |
| **Net arch** | [512,512,512] | **[512,512,256]** (optimal) |
| **N envs** | 64 | **32** (stabilité) |
| **Timesteps** | 10M | **5M** (suffisant) |
| **Commission** | 0.01% | **0.1%** (réaliste) |
| **Trades/jour** | 30 | **20** (conservative) |

---

## 🚀 UTILISATION

### **Installation** 

```bash
cd /root/ai-factory/tmp/project_ploutos
git pull origin main
source /root/ai-factory/venv/bin/activate

# Vérifier GPU
nvidia-smi
```

---

### **Entraînement Standard (Recommandé)** ⭐

```bash
nohup python3 scripts/train_v4_optimal.py \
  --config optimal \
  --wandb \
  --project Ploutos_V4_FINAL \
  > logs/train_v4.log 2>&1 &

# Suivre logs
tail -f logs/train_v4.log
```

**Durée** : 8-10h sur RTX 3080

---

### **Entraînement Rapide (Test)** 🐎

```bash
python3 scripts/train_v4_optimal.py \
  --config fast \
  --output models/test_v4.zip
```

**Config Fast** :
- 2M timesteps (au lieu de 5M)
- 16 envs (au lieu de 32)
- Durée : 3-4h

---

### **Entraînement Qualité Max** 🏆

```bash
nohup python3 scripts/train_v4_optimal.py \
  --config quality \
  --wandb \
  > logs/train_v4_quality.log 2>&1 &
```

**Config Quality** :
- 10M timesteps
- 48 envs
- Durée : 15-18h

---

### **Tickers Custom**

```bash
python3 scripts/train_v4_optimal.py \
  --tickers AAPL MSFT GOOGL NVDA TSLA SPY QQQ \
  --wandb
```

---

## 📊 MONITORING

### **Pendant Entraînement**

```bash
# Logs temps réel
tail -f logs/train_v4.log

# GPU usage
watch -n 5 nvidia-smi

# Processes
ps aux | grep train_v4
```

---

### **W&B Dashboard**

```
https://wandb.ai/vimif-perso/Ploutos_V4_FINAL
```

**Métriques à surveiller** :
- `train/reward` : Doit augmenter
- `eval/mean_reward` : Validation (important !)
- `train/learning_rate` : Doit décroitre
- `time/fps` : FPS stable ~150-200

---

## ✅ INDICATEURS DE SUCCÈS

### **Pendant Entraînement**

```bash
✅ Reward train augmente régulièrement
✅ Reward validation suit (pas trop d'écart)
✅ "NOUVEAU MEILLEUR MODÈLE" toutes les 50-100k steps
✅ Early stopping NE se déclenche PAS avant 3-4M steps
✅ GPU usage 80-95%
```

### **Signes de Problème** ⚠️

```bash
❌ Reward train stagne ou diminue
❌ Reward validation << reward train (overfit)
❌ "Pas d'amélioration" trop tôt (<2M steps)
❌ GPU usage <50%
❌ FPS <100
```

---

## 🎯 OBJECTIFS V4

| Métrique | V3 FIXED Cible | **V4 OPTIMAL Cible** |
|----------|----------------|----------------------|
| **Score 90j** | >90 | **>92** ⭐ |
| **Score 365j** | >80 | **>85** ⭐ |
| **Return 90j** | >50% | **>60%** |
| **Return 365j** | >20% | **>25%** |
| **Drawdown** | <8% | **<6%** ⭐ |
| **Trades/jour** | <30 | **15-25** ⭐ |
| **Win rate** | >55% | **>58%** ⭐ |
| **Sharpe** | >1.5 | **>2.0** ⭐ |

---

## 💾 FICHIERS GÉNÉRÉS

```
models/
  ploutos_v4_optimal.zip      # Modèle final
  ploutos_v4_optimal.json     # Config
  best_model.zip              # ⭐ MEILLEUR modèle (utilise celui-ci !)
  best_metrics.json           # Métriques du meilleur
  checkpoints/
    ploutos_v4_optimal_100000_steps.zip
    ploutos_v4_optimal_200000_steps.zip
    ...
```

**⚠️ IMPORTANT** : Utilise **`best_model.zip`**, pas le final !

---

## 🧪 BACKTEST

### **Après Entraînement**

```bash
# Backtest 90j (rapide)
python3 scripts/backtest_reliability.py \
  --model models/best_model.zip \
  --days 90 \
  --episodes 5

# Backtest 365j (complet)
python3 scripts/backtest_reliability.py \
  --model models/best_model.zip \
  --days 365 \
  --episodes 10
```

**Critères Validation** :
```
✅ Score 90j > 92
✅ Score 365j > 85
✅ Return 365j > 25%
✅ Drawdown < 6%
✅ Trades/jour 15-25
```

---

## 🛠️ TROUBLESHOOTING

### **"Out of Memory" (GPU)**

```bash
# Réduire envs
python3 scripts/train_v4_optimal.py --config fast  # 16 envs au lieu de 32

# Ou modifier directement
# Dans train_v4_optimal.py ligne 46:
'n_envs': 16,  # Au lieu de 32
```

---

### **"Pas d'amélioration" trop tôt**

```python
# Augmenter patience
# Ligne 52:
'patience': 10,  # Au lieu de 5
```

---

### **Training trop lent (FPS <100)**

```bash
# Vérifier GPU usage
nvidia-smi

# Si GPU pas utilisé :
export CUDA_VISIBLE_DEVICES=0

# Ou forcer CPU (plus lent mais marche)
python3 scripts/train_v4_optimal.py --device cpu
```

---

### **W&B ne se connecte pas**

```bash
# Login W&B
wandb login

# Ou désactiver
python3 scripts/train_v4_optimal.py  # Sans --wandb
```

---

## 📚 RÉFÉRENCES

### **Recherche Académique**

1. **PPO** : [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
2. **Spinning Up** : [OpenAI Deep RL Guide](https://spinningup.openai.com/)
3. **RL Trading** : [Deep Reinforcement Learning for Trading](https://arxiv.org/abs/1911.10107)

### **Best Practices**

- Batch size petit (64) pour généralisation
- LR decay pour convergence fine
- Train/val split pour détecter overfit
- Early stopping pour gagner temps
- Commission réaliste (0.1%) pour réalisme

---

## 🏆 CHECKLIST COMPLÈTE

### **Avant Entraînement**

- [ ] GPU fonctionnel (`nvidia-smi`)
- [ ] Données disponibles (10 tickers minimum)
- [ ] Espace disque >10GB
- [ ] W&B login (optionnel)
- [ ] Logs directory existe

### **Pendant Entraînement**

- [ ] GPU usage 80-95%
- [ ] FPS stable 150-200
- [ ] Reward augmente
- [ ] "NOUVEAU MEILLEUR MODÈLE" régulier
- [ ] Pas crash

### **Après Entraînement**

- [ ] `best_model.zip` existe
- [ ] `best_metrics.json` existe
- [ ] Backtest 90j score >92
- [ ] Backtest 365j score >85
- [ ] Trades/jour 15-25
- [ ] Drawdown <6%

---

## 🚀 QUICK START

```bash
# Clone
cd /root/ai-factory/tmp/project_ploutos
git pull

# Activate
source /root/ai-factory/venv/bin/activate

# Train
nohup python3 scripts/train_v4_optimal.py \
  --wandb \
  --project Ploutos_V4_FINAL \
  > logs/train_v4.log 2>&1 &

# Monitor
tail -f logs/train_v4.log

# Wait 8-10h...

# Backtest
python3 scripts/backtest_reliability.py \
  --model models/best_model.zip \
  --days 365 \
  --episodes 10

# Si OK :
# → Déployer sur VPS
# → Paper trading 7 jours
# → LIVE 🚀
```

---

**Date** : 9 Décembre 2025  
**Version** : V4 OPTIMAL  
**Status** : ✅ PRÊT À UTILISER  
**Auteur** : Ploutos AI Team
