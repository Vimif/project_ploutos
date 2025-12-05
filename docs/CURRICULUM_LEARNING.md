# 🎓 Curriculum Learning pour Ploutos

## 📊 Vue d'ensemble

Le **Curriculum Learning** est une approche d'entraînement progressif qui permet au modèle d'apprendre de manière incrémentale, du plus simple au plus complexe.

### 🎯 Objectifs

- ✅ **Améliorer la robustesse** : Le modèle apprend à généraliser progressivement
- ✅ **Transfer Learning** : Chaque stage bénéficie du précédent
- ✅ **Sharpe Ratio supérieur** : Attendu entre 1.8 et 2.5 (vs 1.0-1.5 classique)
- ✅ **Auto-optimisation optionnelle** : 15 trials Optuna pour affiner les hyperparams

---

## 🔄 Les 3 Stages

### **Stage 1 : Mono-Asset (SPY)**

```yaml
Objectif    : Apprendre les bases sur 1 ETF simple
Asset       : SPY uniquement
Timesteps   : 3,000,000
N_envs      : 4
Durée       : ~3 heures (RTX 3080)
Sharpe cible: 1.0
```

**Pourquoi SPY ?**
- ETF diversifié (S&P 500), peu volatil
- Mouvements prévisibles et structurés
- Parfait pour apprendre les bases du trading

---

### **Stage 2 : Multi-Asset (ETFs)**

```yaml
Objectif    : Généraliser sur plusieurs ETFs
Assets      : SPY, QQQ, IWM
Timesteps   : 5,000,000
N_envs      : 8
Durée       : ~5 heures
Sharpe cible: 1.3
```

**Nouvelles compétences :**
- Gérer un portfolio multi-asset
- Corrélations entre ETFs
- Allocation dynamique

---

### **Stage 3 : Actions Complexes**

```yaml
Objectif    : Maîtriser des actions volatiles
Assets      : NVDA, MSFT, AAPL, GOOGL, AMZN
Timesteps   : 10,000,000
N_envs      : 16
Durée       : ~10 heures
Sharpe cible: 1.5
```

**Défis supplémentaires :**
- Volatilité élevée (NVDA peut varier de 5-10% par jour)
- Corrélations complexes (tech stocks)
- Risque de drawdown accru

---

## 🚀 Utilisation

### **Option 1 : Sans Auto-Optimisation (Rapide)**

```bash
cd /root/ai-factory/tmp/project_ploutos

# Stage 1
python3 scripts/train_curriculum.py --stage 1

# Stage 2 (avec transfer learning automatique)
python3 scripts/train_curriculum.py --stage 2

# Stage 3
python3 scripts/train_curriculum.py --stage 3

# Durée totale : ~18 heures
```

---

### **Option 2 : Avec Auto-Optimisation (Optimal)**

```bash
# Stage 1 avec optimisation rapide (15 trials)
python3 scripts/train_curriculum.py --stage 1 --auto-optimize

# Stage 2
python3 scripts/train_curriculum.py --stage 2 --auto-optimize

# Stage 3
python3 scripts/train_curriculum.py --stage 3 --auto-optimize

# Durée totale : ~20 heures (+30min par stage pour Optuna)
# Gain Sharpe attendu : +0.2 à +0.3 par stage
```

---

### **Option 3 : Stage par Stage Manuel**

```bash
# Entraîner Stage 1
python3 scripts/train_curriculum.py --stage 1 --auto-optimize

# Attendre fin, vérifier Sharpe > 1.0

# Passer à Stage 2 avec modèle Stage 1
python3 scripts/train_curriculum.py --stage 2 \
  --load-model models/stage1_final \
  --auto-optimize

# Attendre fin, vérifier Sharpe > 1.3

# Passer à Stage 3
python3 scripts/train_curriculum.py --stage 3 \
  --load-model models/stage2_final \
  --auto-optimize
```

---

## 📊 Suivi de l'Entraînement

### **Weights & Biases**

```bash
# Le script log automatiquement sur W&B
# Projet : Ploutos_Curriculum

# Accéder au dashboard :
https://wandb.ai/your-username/Ploutos_Curriculum
```

**Métriques trackées :**
- Loss (policy + value)
- Sharpe Ratio
- Portfolio Value
- Success Rate

---

### **TensorBoard (Local)**

```bash
# Lancer TensorBoard
tensorboard --logdir logs/

# Accéder à : http://localhost:6006
```

---

### **GPU Monitoring**

```bash
# Surveiller l'utilisation GPU
watch -n 1 nvidia-smi

# Ou avec gpustat (plus lisible)
pip install gpustat
gpustat -i 1
```

---

## 💾 Modèles Sauvegardés

```
models/
├── stage1/
│   ├── ploutos_stage1_50000_steps.zip
│   ├── ploutos_stage1_100000_steps.zip
│   ├── optimized_params.json           # Si --auto-optimize
│   └── ...
├── stage1_final.zip                   # Modèle final Stage 1
├── stage2/
│   ├── ploutos_stage2_100000_steps.zip
│   ├── optimized_params.json
│   └── ...
├── stage2_final.zip                   # Modèle final Stage 2
├── stage3/
│   ├── ploutos_stage3_200000_steps.zip
│   ├── optimized_params.json
│   └── ...
└── stage3_final.zip                   # 🎯 Modèle PRODUCTION
```

---

## ⚡ Auto-Optimisation Rapide

### **Fonctionnement**

Quand `--auto-optimize` est activé :

1. **15 trials Optuna** (au lieu de 50 classiques)
2. **Optimise seulement 3 params critiques** :
   - `learning_rate`
   - `n_steps`
   - `ent_coef`
3. **Durée** : +30 minutes par stage
4. **Gain Sharpe** : +0.2 à +0.3

### **Params Pré-Calibrés vs Optimisés**

| Stage | Learning Rate (Base) | Learning Rate (Optimisé) | Gain Sharpe |
|-------|---------------------|--------------------------|-------------|
| 1     | 1e-4               | 8e-5 à 2e-4            | +0.2        |
| 2     | 5e-5               | 3e-5 à 1e-4            | +0.3        |
| 3     | 3e-5               | 1e-5 à 6e-5            | +0.3        |

---

## 📈 Résultats Attendus

### **Sans Auto-Optimisation**

```
Stage 1 : Sharpe = 1.0-1.2
Stage 2 : Sharpe = 1.3-1.5
Stage 3 : Sharpe = 1.5-1.8

Durée totale : 18h
```

### **Avec Auto-Optimisation**

```
Stage 1 : Sharpe = 1.2-1.4  (+0.2)
Stage 2 : Sharpe = 1.6-1.8  (+0.3)
Stage 3 : Sharpe = 1.8-2.3  (+0.3)

Durée totale : 20h (+2h Optuna)
```

---

## 🛠️ Dépannage

### **Erreur : CUDA out of memory**

```bash
# Réduire batch_size et n_envs dans le code
# Ou redémarrer le GPU
sudo systemctl restart nvidia-persistenced
```

### **Erreur : SubprocVecEnv freeze**

```bash
# Utiliser DummyVecEnv si problèmes multiprocessing
# Modifier dans train_curriculum.py ligne 389 :
from stable_baselines3.common.vec_env import DummyVecEnv
env = DummyVecEnv([make_env(data)])
```

### **Sharpe Ratio < 0**

```
Causes possibles :
1. Données corrompues (vérifier data_cache/)
2. Modèle précédent incompatible
3. Hyperparams inadaptés

Solution :
- Nettoyer data_cache/
- Repartir de zéro sans --load-model
- Activer --auto-optimize
```

---

## 💡 Bonnes Pratiques

### **1. Lancer en Arrière-Plan**

```bash
# Avec nohup
nohup python3 scripts/train_curriculum.py --stage 1 --auto-optimize \
  > logs/stage1_$(date +%Y%m%d_%H%M).log 2>&1 &

# Suivre les logs
tail -f logs/stage1_*.log
```

### **2. Vérifier GPU Disponible**

```bash
# Avant de lancer
nvidia-smi

# Si GPU utilisé par autre process, tuer ou attendre
```

### **3. Sauvegarder Régulièrement**

```bash
# Les checkpoints sont auto-sauvegardés tous les N steps
# Mais copier le dernier modèle après chaque stage :
cp models/stage1_final.zip backups/stage1_$(date +%Y%m%d).zip
```

---

## 🔗 Ressources

- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Weights & Biases](https://wandb.ai/)
- [PPO Algorithm Explained](https://arxiv.org/abs/1707.06347)

---

## 🎯 Prochaines Étapes

Après avoir terminé le curriculum learning :

1. **Évaluer le modèle** sur données de validation
2. **Déployer sur VPS** pour trading live
3. **Créer des modèles spécialisés par régime** (Bull/Bear/Sideways)
4. **Implémenter le ré-entraînement automatique** mensuel

---

**Questions ou problèmes ?** Ouvrir une issue sur GitHub ou consulter les logs.