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

## Notes de Migration

- Les modeles entraines avec V8 sont compatibles V9 (l'environnement a la meme signature d'observation).
- L'utilisation de Shared Memory est automatique si activee dans `config.yaml` ou via `--shared-memory`.

---

# Release Notes V9.1: The "Reliability" Update

**Date:** 14 Fevrier 2026
**Focus:** Robustesse, refactoring, tests, production fixes

## Highlights

### 1. Refactoring Environnement (Component Architecture)
- **`core/env_config.py`** : `EnvConfig` dataclass structuree avec `TransactionConfig`, `RewardConfig`, `TradingConfig`. Remplace 30+ parametres du constructeur. `from_flat_dict()` pour compatibilite YAML.
- **`core/reward_calculator.py`** : Calcul DSR extrait avec algorithme de Welford (variance online, numeriquement stable). Penalties drawdown et overtrading.
- **`core/observation_builder.py`** : Construction du vecteur d'observation extraite (features + macro + portfolio state).
- **`core/constants.py`** : Constantes centralisees (plus de magic numbers).
- **`core/exceptions.py`** : `PloutosError` -> `ConfigValidationError`, `DataFetchError`, `TrainingError`, `InsufficientDataError`.

### 2. Fix Critique : Look-Ahead Bias
- Les features sont desormais calculees **par fold** dans la boucle walk-forward.
- Avant : le "Turbo Init" pre-calculait sur TOUT le dataset (6 ans) avant les splits, injectant de l'info future dans les periodes de test.

### 3. Fix Production : Crash Timestamp
- **Probleme** : L'evaluation sur le test set crashait avec `TypeError: float() argument must be a string or a real number, not 'Timestamp'`.
- **Cause racine** : Triple bug :
  1. Le filtre dtype `(np.float64, np.float32, np.int64)` excluait `int32` (produit par Polars `cast(pl.Int32)` pour ~25 features binaires). Le chemin SHM convertissait tout en float32, masquant le probleme en training.
  2. La conversion Polars ne restaurait pas l'index si son nom ne matchait pas exactement 'date'/'time'/'index' (case-sensitive).
  3. `fill_null(0)` sur une colonne datetime pouvait la corrompre.
- **Fix** : `pd.api.types.is_numeric_dtype()`, normalisation index `__date_idx`, protection datetime pendant cleanup.

### 4. Couts de Transaction Realistes
- `spread_bps` : 2.0 -> 5.0 (plus realiste en conditions de marche).
- `market_impact_factor` : 0.0001 -> 0.00015.
- `vol_ceiling` configurable (remplace le 0.05 hardcode).

### 5. Observation Space Reduction
- `max_features_per_ticker: 30` dans config.yaml.
- Selection par variance (top-N features par ticker).
- `net_arch: [256, 128]` au lieu de `[512, 512, 512]` (ratio parametres/samples ameliore).

### 6. Robustesse renforcee
- Bruit Monte Carlo : 0.1% -> 0.5% (5x, plus realiste).
- Seuil overfitting : 5% -> 20% (moins de faux positifs).
- Crash instantane ajoute aux stress tests.
- Ensemble : `predict_filtered()` avec filtrage par confiance.

### 7. Tests & CI/CD
- **116 tests** couvrant : env, reward calculator, config, transaction costs, ensemble, features, shared memory, integration.
- **GitHub Actions** : pytest + black + ruff + mypy, matrice Python 3.10/3.11, coverage enforcement.
- **Config validation** : `config/schema.py` etendu avec sections network/checkpoint/eval + contraintes croisees.

## Nouveaux Modules
- `core/env_config.py`
- `core/reward_calculator.py`
- `core/observation_builder.py`
- `core/constants.py`
- `core/exceptions.py`
- `tests/test_env_config.py` (11 tests)
- `tests/test_reward_calculator.py` (9 tests)
- `tests/test_transaction_costs.py` (13 tests)

## Breaking Changes
- `net_arch` default : `[512, 512, 512]` -> `[256, 128]`
- `spread_bps` default : `2.0` -> `5.0`
- `use_shared_memory` default : `false` -> `true`
- Les modeles V9.0 entraines avec l'ancien observation space (1318 dims) ne sont **pas compatibles** avec V9.1 (dims reduits via `max_features_per_ticker`).

---
*Derniere mise a jour : Fevrier 2026*
