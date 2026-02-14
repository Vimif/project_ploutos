# Ploutos Developer Knowledge & Troubleshooting

Ce document consigne les leçons apprises, les problèmes techniques résolus et les optimisations spécifiques à ce projet. À lire avant toute modification majeure.

---

## 1. Données Historiques (1h Timeframe)

### Problème : Limite Yahoo Finance
- **Limite** : Yahoo Finance ne fournit que **730 jours (2 ans)** d'historique pour les données horaires (`1h`).
- **Conséquence** : Impossible d'entraîner un modèle robuste sur 5 ans avec Yahoo seul.

### Solution : Alpaca + Local Dataset
- Utiliser l'API **Alpaca** (Paper ou Live) qui offre **5+ ans** d'historique en `1h`.
- Script : `scripts/build_dataset.py` télécharge tout en CSV dans `data/dataset_v8/`.
- Config : `training_config_v8.yaml` -> `dataset_path: ./data/dataset_v8/`.
- **Règle d'Or** : Toujours utiliser le dataset local pour l'entraînement intensif. Ne plus télécharger à la volée.

---

## 2. Conflits de Timezones (Pandas)

### Problème : `TypeError: Cannot compare dtypes datetime64[s] and datetime64[us, UTC]`
- **Cause** : 
    - Données Alpaca (Local Dataset) sont en **UTC** (tz-aware).
    - Données Macro (Yahoo) sont souvent **naïves** (tz-naive) ou vice-versa.
    - Lors du `reindex` dans `core/macro_data.py`, Pandas crashe.

### Solution : Alignement Automatique
- Dans `core/macro_data.py` -> `align_to_ticker()`.
- Le code détecte la timezone du ticker cible et convertit la macro pour qu'elle corresponde (soit `tz_localize('UTC')`, soit `tz_localize(None)`).

---

## 3. Optimisation Hardware (RunPod / Serveurs Dédiés)

### Contexte : Machine "Monstre" (256 Cores, 1TB RAM, RTX 3090)
Les paramètres par défaut de `stable-baselines3` sous-utilisent massivement ce genre de hardware (10-20% de charge).

### Paramètres "Turbo" (dans `config/hardware.py`)
Pour saturer la machine et accélérer l'entraînement :
- **n_envs** : Monter à **256** (1 env par CPU core).
- **batch_size** : Monter à **65536** (pour remplir les 24GB VRAM de la 3090).
- **n_steps** : Monter à **4096** (réduit l'overhead CPU/GPU).
- **Ensemble** : Lancer 10 ou 20 modèles en parallèle (`--ensemble 20`).

### Risques
- **OOM (Out Of Memory)** : Si 256 envs consomment plus que la RAM dispo (peu probable avec 1TB, mais possible).
- **Latence de démarrage** : Initialiser 256 environnements prend du temps (1-2 minutes).

---

## 4. Stratégie Walk-Forward & Overfitting

### Constat (Février 2026)
- Entraîner sur 1 an -> Overfitting massif (25% pertes en Monte Carlo).
- Le modèle "apprend par cœur" le bruit de marché.

### Correction
- **Fenêtre d'entraînement** : Minimum **5 ans** (`train_years: 5`).
- **Simplicité** : Réseau de neurones plus large mais moins profond (`[512, 512, 512]` au lieu de 4 couches).
- **Moins d'aléatoire** : `ent_coef` réduit à `0.01` (vs 0.05).
- **Ensemble Learning** : Toujours utiliser plusieurs modèles (`--ensemble 5+`) pour lisser la variance.
