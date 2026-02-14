# Documentation Architecture V9 (Ploutos Ultimate)

## 📌 Vue d'Ensemble
L'architecture **V9** (Février 2026) est une refonte majeure de l'infrastructure de traitement de données et d'entraînement pour permettre le **Massively Parallel Training** sur des architectures multi-cœurs (RunPod/Cloud GPU).

Elle s'appuie sur deux piliers principaux :
1. **Polars High-Performance Engine** : Calcul des indicateurs techniques ultra-rapide.
2. **Shared Memory Manager** : Partage de données zéro-copie entre processus d'entraînement.

## 🏗️ Structure des Composants

### 1. Feature Engineering (`core/features.py`)
- **Backend** : Polars (`pl.DataFrame`) pour les calculs vectorisés (SIMD).
- **Frontend** : Pandas (`pd.DataFrame`) pour la compatibilité avec `gym` et `stable-baselines3`.
- **Workflow** :
  - Input: Pandas DataFrame (OHLCV).
  - Processing: Conversion LazyFrame -> Calculs (85+ features) -> Collect -> Conversion Pandas.
  - Output: Pandas DataFrame enrichi.
- **Performance** : x100 par rapport à Pandas pur.

### 2. Environment (`core/environment.py`)
Le nouvel environnement `TradingEnv` unifié remplace les anciennes versions (V6, V8, Universal).
- **Compatible** : PPO (MLP) et RecurrentPPO (LSTM).
- **Mémoire Partagée** : Si activé, lit les données directement depuis la RAM partagée (`SharedMemory`) au lieu de copier les DataFrames pour chaque worker.
- **Macro Data** : Intègre nativement les données macroéconomiques (VIX, TNX, DXY).

### 3. Training Pipeline (`training/train.py`)
Le script principal d'entraînement a été refondu pour gérer efficacement les ressources.
- **Auto-Scaling** : Détecte le matériel (CPU cores, RAM) et ajuste `n_envs` automatiquement.
- **Shared Memory Auto-Init** : Lance le `SharedDataManager` avant de forker les processus SubprocVecEnv.

### 4. Shared Data Manager (`core/shared_memory_manager.py`)
- Utilise `multiprocessing.shared_memory` pour stocker les arrays Numpy (Open, High, Low, Close, Volume, Features) dans un segment mémoire unique.
- Les workers accèdent aux données en lecture seule via des pointeurs, évitant la duplication mémoire (Zero-Copy).

## 🚀 Flux de Données (Data Flow)

1. **Download** (`DataFetcher`) -> Pandas DF (RAM locale).
2. **Feature Engineering** (`FeatureEngineer`) -> Polars -> Pandas DF (RAM locale).
3. **Shared Memory Put** (`SharedDataManager`) -> Copie vers SharedMemory Block.
4. **Environment Init** (`TradingEnv`) -> Reçoit le nom du bloc SharedMemory.
5. **Step()** : L'environnement lit les données via `np.ndarray` mappé sur la mémoire partagée.

Même avec 64 workers, la consommation mémoire reste celle d'une seule copie des données !

## 📁 Organisation des Fichiers

```text
/project_ploutos
├── core/
│   ├── environment.py       # V9 Environment
│   ├── features.py          # Polars Feature Engine
│   ├── shared_memory_manager.py # Shared Memory Logic
│   └── ...
├── training/
│   └── train.py             # V9 Training Script
├── legacy/                  # Archives (V6, V7, V8)
└── scripts/
    ├── validate_pipeline.py # Validation V9
    └── paper_trade.py       # Paper Trading V9
```
