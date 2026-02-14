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
- **Latence de démarrage** : Initialiser 256 environnements prend du temps (1-2 minutes).

### 🚨 Thread Explosion (OpenBLAS / MKL)
- **Symptôme** : `OpenBLAS blas_thread_init: pthread_create failed`, `Resource temporarily unavailable`, `BrokenPipeError`.
- **Cause** : Si on lance 256 processus (`n_envs`) et que chaque processus lance 128 threads (OpenBLAS par défaut), on atteint **32 768 threads**. Linux tue le programme.
- **Solution** : Forcer 1 thread par processus via variables d'env :
  ```bash
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  ```
- **Script** : Utiliser `./start_training.sh` qui configure tout cela automatiquement.

### 🚨 RAM Limitation (116 GB)
- **Symptôme** : Crash "Out Of Memory" (OOM) au démarrage des 256 environnements.
- **Cause** : Chaque env consomme ~400Mo. 256 * 400Mo = ~100Go + Système = Saturation.
- **Solution** : Le script `config/hardware.py` limite désormais automatiquement `n_envs` à **~90-128** si < 128GB RAM détectés.

---

## 4. Stratégie Walk-Forward & Overfitting

### Constat (Février 2026)
- Entraîner sur 1 an -> Overfitting massif (25% pertes en Monte Carlo).
- Le modèle "apprend par cœur" le bruit de marché.

### Correction
- **Fenêtre d'entraînement** : Minimum **5 ans** (`train_years: 5`).
- **Simplicite** : Reseau `[256, 128]` au lieu de `[512, 512, 512]` (meilleur ratio parametres/samples).
- **Moins d'aléatoire** : `ent_coef` réduit à `0.01` (vs 0.05).
- **Ensemble Learning** : Toujours utiliser plusieurs modèles (`--ensemble 5+`) pour lisser la variance.

---

## 5. Optimisation : "Turbo Init" (Pre-computed Features)

### Problème : Démarrage Lent des Environnements
- Avec `n_envs=256`, chaque processus calcule indépendamment les indicateurs techniques (RSI, MACD, etc.) pour 15 tickers sur 50 000 bougies.
- **Coût** : 256 processus x 15 tickers x 50k bougies = **192 Millions de calculs** au démarrage.
- **Résultat** : CPU à 100% pendant des minutes, risque de timeout ou crash.

### Solution : Calcul par Fold & Partage
- **V9.1** : `training/train.py` calcule les features **par fold** (pas sur tout le dataset) pour eviter le look-ahead bias.
- **Injection** : Les DataFrames enrichis sont passes aux environnements avec le flag `features_precomputed=True`.
- **Environnement** : `TradingEnv` detecte le flag et **saute** le calcul interne.
- **Gain** : Demarrage quasi-instantane des 256 environnements (juste copie memoire).

---

## 6. Méthodologie Institutionnelle (V8.1 - Février 2026)

### 🛡️ Embargo (Anti-Leak)
- **Problème** : Les indicateurs techniques (ex: EMA 200, RSI 14) "regardent en arrière". Si le Test Set commence immédiatement après le Train Set, les premières 200 bougies de Test contiennent de l'information déjà vue par le Train (Data Leakage).
- **Solution** : `training/train.py` impose un **Embargo** (gap) de 1 mois entre la fin du Train et le début du Test.
- **Impact** : Performance Test légèrement moins bonne MAIS beaucoup plus réaliste.

### Differential Sharpe Ratio (DSR)
- **Probleme** : Recompenser le Profit ($) incite a la prise de risque excessive (gambling).
- **Solution** : `core/reward_calculator.py` implemente le **DSR** avec l'algorithme de **Welford** pour la variance online (ref: Moody & Saffell, 2001).
- **Principe** : L'agent est recompense si son action augmente le Sharpe Ratio glissant (Risk-Adjusted Return) plutot que le PnL brut.
- **Stabilite** : Welford evite l'explosion numerique quand `std(returns) ≈ 0` (marche plat). Floor de variance a `1e-4`.

### Hyperparametres PPO "Investisseur"
- **GAE Lambda** : Augmente a **0.98** (vs 0.95) pour favoriser les tendances long terme et reduire le bruit.
- **Overtrading Penalty** : Doublee (`0.01`) pour punir severement le "churning" (achat/vente inutile).
- **Trade Success Reward** : Reduite (`0.2`) pour ne pas biaiser l'agent vers des strategies a haut taux de reussite mais faible gain moyen.

---

## 7. Divergence SHM / Raw Data Path (V9.1 - Fevrier 2026)

### Probleme : Train/Test Feature Set Mismatch
- **Symptome** : `TypeError: float() argument must be a string or a real number, not 'Timestamp'` lors de `evaluate_on_test`.
- **Cause racine** : Le chemin training (SharedMemory) et le chemin test (DataFrames bruts) produisaient des feature sets differents.
  - SHM : `select_dtypes(include=[np.number]).astype(float32)` homogeneise tous les types numeriques (int32 -> float32).
  - Raw : le filtre dtype `(np.float64, np.float32, np.int64)` excluait les colonnes `int32` produites par Polars `cast(pl.Int32)` (~25 features binaires).
- **Fix** : Remplacer le filtre explicite par `pd.api.types.is_numeric_dtype()` dans `_prepare_features()`.

### Probleme : Polars Index Round-Trip
- **Symptome** : Colonne datetime residuelle dans le DataFrame apres conversion Polars -> Pandas.
- **Cause** : Le nom d'index varie ('Date', 'Datetime', None) et la logique de restauration `set_index('date')` est case-sensitive.
- **Fix** : Normaliser l'index en `__date_idx` avant conversion, restaurer apres.

### Regle d'Or
Toujours verifier la parite entre le chemin SHM (training) et le chemin raw (test). Si le test crash mais le training fonctionne, le probleme est probablement une divergence dtype.

---

## 8. Tests & Mocking (V9.1 - Fevrier 2026)

### Mocking torch/SB3 dans les tests
- Mocker TOUS les sous-modules torch : `torch.nn`, `torch.nn.functional`, `torch.optim`, `torch.utils`, `torch.utils.data`, `torch.distributions`.
- Aussi : `stable_baselines3`, `stable_baselines3.common`, `stable_baselines3.common.vec_env`, etc.
- **NE PAS** mettre le mock torch dans `conftest.py` — ca casse les tests e2e qui ont besoin du vrai torch.
- `isinstance()` crashe si le 2e argument est un MagicMock (pas un type) — utiliser `HAS_RECURRENT=False` + `patch.object()`.

### Tests de look-ahead bias
- **NE PAS** generer deux datasets separes avec `np.random.seed(42)` — l'etat RNG diverge apres des generations de longueurs differentes.
- Generer UN seul dataset long, le decouper pour la version courte : `df_short = df_long.iloc[:200].copy()`.

