# Documentation Architecture V9.1 (Ploutos Ultimate)

## Vue d'Ensemble
L'architecture **V9.1** (Fevrier 2026) est une refonte majeure de l'infrastructure de traitement de donnees et d'entrainement pour permettre le **Massively Parallel Training** sur des architectures multi-coeurs (RunPod/Cloud GPU).

Elle s'appuie sur quatre piliers principaux :
1. **Polars High-Performance Engine** : Calcul des indicateurs techniques ultra-rapide.
2. **Shared Memory Manager** : Partage de donnees zero-copie entre processus d'entrainement.
3. **Component Architecture** : Environnement decoupe en sous-composants (RewardCalculator, ObservationBuilder, EnvConfig).
4. **Config Validation & CI/CD** : Schema de validation, exceptions custom, 116 tests, GitHub Actions.

## Structure des Composants

### 1. Feature Engineering (`core/features.py`)
- **Backend** : Polars (`pl.DataFrame`) pour les calculs vectorises (SIMD).
- **Frontend** : Pandas (`pd.DataFrame`) pour la compatibilite avec `gym` et `stable-baselines3`.
- **Index Round-Trip** : Normalise l'index en `__date_idx` pour un aller-retour fiable Pandas->Polars->Pandas, quel que soit le nom d'index original ('Date', 'Datetime', None, etc.).
- **Workflow** :
  - Input: Pandas DataFrame (OHLCV + DatetimeIndex).
  - Processing: Conversion Polars -> Calculs (60+ features) -> Cleanup (fill_nan/fill_null) -> Conversion Pandas.
  - Output: Pandas DataFrame enrichi avec DatetimeIndex restaure.
- **Performance** : x100 par rapport a Pandas pur.

### 2. Environment (`core/environment.py`)
Le nouvel environnement `TradingEnv` unifie remplace les anciennes versions (V6, V8, Universal).
- **Compatible** : PPO (MLP) et RecurrentPPO (LSTM).
- **Memoire Partagee** : Si active, lit les donnees directement depuis la RAM partagee (`SharedMemory`) au lieu de copier les DataFrames pour chaque worker.
- **Macro Data** : Integre nativement les donnees macroeconomiques (VIX, TNX, DXY).
- **Feature Selection** : `max_features_per_ticker` selectionne les top-N features par variance pour reduire l'observation space.
- **Dtype Safety** : Utilise `pd.api.types.is_numeric_dtype()` pour filtrer les colonnes features + `pd.to_numeric(errors="coerce")` comme filet de securite.

### 2b. Composants extraits (V9.1)
L'environnement a ete decompose en sous-composants pour la maintenabilite :
- **`core/env_config.py`** : `EnvConfig` dataclass structuree (TransactionConfig, RewardConfig, TradingConfig). Remplace les 30+ parametres du constructeur. `from_flat_dict()` pour compatibilite YAML.
- **`core/reward_calculator.py`** : Calcul DSR (Differential Sharpe Ratio) avec algorithme de Welford pour la variance online. Penalties drawdown et overtrading.
- **`core/observation_builder.py`** : Construction du vecteur d'observation (features + macro + portfolio state). Clipping et normalisation.
- **`core/constants.py`** : Constantes centralisees (`OBSERVATION_CLIP_RANGE`, `DSR_VARIANCE_FLOOR`, `BANKRUPTCY_THRESHOLD`, `MAX_REWARD_CLIP`, `PORTFOLIO_HISTORY_WINDOW`, etc.).
- **`core/exceptions.py`** : Hierarchie d'exceptions custom (`PloutosError` -> `ConfigValidationError`, `DataFetchError`, `TrainingError`, `InsufficientDataError`).

### 3. Training Pipeline (`training/train.py`)
Le script principal d'entrainement a ete refondu pour gerer efficacement les ressources.
- **Auto-Scaling** : Detecte le materiel (CPU cores, RAM, GPU) et ajuste `n_envs` automatiquement.
- **Shared Memory Auto-Init** : Lance le `SharedDataManager` avant de forker les processus SubprocVecEnv.
- **Per-Fold Feature Computation** : Les features sont calculees par fold pour eviter le look-ahead bias.
- **Config Validation** : `config/schema.py` valide la config au chargement (types, ranges, contraintes croisees `n_envs * n_steps >= batch_size`).

### 4. Shared Data Manager (`core/shared_memory_manager.py`)
- Utilise `multiprocessing.shared_memory` pour stocker les arrays Numpy en float32 dans un segment memoire unique.
- Les workers accedent aux donnees en lecture seule via des pointeurs (Zero-Copy).
- **Attention** : `put_data()` appelle `select_dtypes(include=[np.number]).astype(float32)` ce qui homogeneise tous les types numeriques. Le chemin test (sans SHM) doit utiliser `is_numeric_dtype()` pour rester coherent.

### 5. Config Validation (`config/schema.py`)
- Valide les sections : `training`, `environment`, `data`, `walk_forward`, `network`, `checkpoint`, `eval`.
- Detecte les typos (cles inconnues).
- Verifie les types et les bornes (min/max).
- Contraintes croisees : `n_envs * n_steps >= batch_size`.
- Leve `ConfigValidationError` (custom exception).

## Flux de Donnees (Data Flow)

```
1. Download (DataFetcher) -> Pandas DF (OHLCV + DatetimeIndex)
2. Walk-Forward Split (generate_walk_forward_splits) -> train/test slices par fold
3. Feature Engineering per fold (FeatureEngineer) -> Polars -> Pandas DF enrichi
4. [Training] Shared Memory Put (SharedDataManager) -> float32 arrays en SHM
5. [Training] TradingEnv init -> load_shared_data() -> DataFrames reconstruits
6. [Test] TradingEnv init -> DataFrames bruts (pas de SHM)
7. _prepare_features() -> is_numeric_dtype() filter -> variance selection -> numpy arrays
```

Meme avec 64 workers, la consommation memoire reste celle d'une seule copie des donnees !

## Organisation des Fichiers

```text
/project_ploutos
├── core/
│   ├── environment.py           # V9 Environment (orchestrateur leger)
│   ├── env_config.py            # EnvConfig dataclass
│   ├── observation_builder.py   # Construction observations
│   ├── reward_calculator.py     # DSR + penalties (Welford)
│   ├── constants.py             # Magic numbers centralises
│   ├── exceptions.py            # Hierarchie d'exceptions custom
│   ├── features.py              # Polars Feature Engine (60+ features)
│   ├── transaction_costs.py     # Slippage/spread/commission (vol_ceiling configurable)
│   ├── ensemble.py              # Multi-model voting + predict_filtered()
│   ├── shared_memory_manager.py # Shared Memory Zero-Copy
│   ├── data_fetcher.py          # Yahoo Finance / Alpaca
│   └── macro_data.py            # VIX/TNX/DXY
├── config/
│   ├── schema.py                # Validation YAML (types + ranges + cross-field)
│   ├── hardware.py              # Auto-detect GPU/CPU/RAM
│   └── config.yaml              # Config principale
├── training/
│   └── train.py                 # V9 Walk-Forward Training
├── tests/                       # 116 tests (pytest)
├── scripts/                     # CLI tools
│   ├── run_pipeline.py          # Pipeline complet
│   ├── robustness_tests.py      # Monte Carlo + stress tests
│   └── paper_trade.py           # Paper Trading
└── .github/workflows/tests.yml  # CI: pytest + black + ruff + mypy
```

---
*Document mis a jour - Fevrier 2026 (V9.1)*
