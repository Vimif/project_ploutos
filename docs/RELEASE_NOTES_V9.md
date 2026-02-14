# Release Notes V9: The "LightSpeed" Update 🚀

**Date:** 14 Février 2026
**Architecture:** V9 Hybrid (Polars + Shared Memory)

## 🌟 Highlights

Cette mise à jour majeure se concentre sur la **performance brute** et l'optimisation des **ressources** pour l'entraînement à grande échelle (RunPod / Multi-GPU).

### 1. Feature Engineering Ultra-Rapide (x100) ⚡
- **Nouveau backend Polars**: Remplacement complet de Pandas par Polars pour le calcul des indicateurs techniques.
- **Gain de performance**: Le calcul des 85+ features sur 100,000 bougies prend maintenant **~0.09s** (contre ~5-10s auparavant).
- **Zéro délai au démarrage**: Le "Turbo Init" pré-calcule tout instantanément.

### 2. Shared Memory Integration (RAM Optimization) 🧠
- **Zéro-Copy Data**: Les workers d'entraînement (SubprocVecEnv) lisent désormais les données directement depuis une mémoire partagée (SharedMemory), sans dupliquer les DataFrames.
- **Gain Mémoire**: Réduction drastique de l'utilisation RAM (par ex. 32 Go -> 4 Go pour 16 environnements).
- **Vitesse**: Plus besoin de sérialiser/désérialiser les données entre processus.

### 3. Architecture Unifiée 🏗️
- **`core/environment.py`**: Devenu le standard unique (remplace `UniversalTradingEnvV6/V8`). Supporte nativement LSTM et Shared Memory.
- **`core/features.py`**: Interface unique compatible Pandas mais propulsée par Polars.
- **Clean Code**: Suppression des anciens environnements V6/V7/V8 (déplacés dans `legacy/`).

## 🛠️ Changements Techniques

### Nouveaux Modules
- `core/features.py`: Moteur de calcul Polars.
- `core/shared_memory_manager.py`: Gestionnaire de mémoire partagée.
- `training/train.py`: Pipeline d'entraînement unifié (anciennement `train_walk_forward.py`).

### Dépendances Ajoutées
- `polars>=0.20.0`
- `pyarrow>=14.0.0`

### Scripts Mis à Jour
- `scripts/validate_pipeline.py`: Utilise maintenant le moteur V9.
- `scripts/paper_trade.py`: Support V9 et détection automatique des modèles.

## ⚠️ Notes de Migration

- Les modèles entraînés avec V8 sont compatibles V9 (l'environnement a la même signature d'observation).
- L'utilisation de Shared Memory est automatique si activée dans `config.yaml` ou via `--shared-memory`.
