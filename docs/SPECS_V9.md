# 🚀 Ploutos V9 : Spécifications Architecturales

**Objectif** : Transformer le prototype de recherche (V8) en un moteur de trading quantitatif industriel, scalable et résilient.

---

## 🏗️ 1. Refonte du Core (Performance & Mémoire)

### 1.1 Shared Memory Architecture (Priorité Critique 🔴)
Actuellement, chaque environnement (128 envs) charge sa propre copie des données, saturant la RAM (116GB+).
**Solution V9** :
- Utiliser `multiprocessing.shared_memory` pour charger le dataset (Numpy array) **une seule fois** en mémoire principale.
- Les environnements accèderont à ces données en **Lecture Seule (Zero-Copy)**.
- **Gain attendu** : Réduction de la RAM de 95% (de 100GB à ~5GB). Capacité de scaler à 256+ cœurs sans OOM.

### 1.2 Polars pour le Data Engineering (Priorité Haute 🟠)
Remplacer `Pandas` par **Polars** pour le pipeline de feature engineering (`FeatureEngineer`).
- **Gain attendu** : Traitement des données 50x plus rapide, lazy evaluation, et meilleure gestion mémoire lors du `Turbo Init`.

---

## 🛡️ 2. Fiabilité & Qualité (CI/CD)

### 2.1 Suite de Tests Unitaires (Tests-First)
Aucun code ne doit être fusionné sans passer une suite de tests.
- **`tests/core/`** : Valider que `UniversalTradingEnv` calcule correctement le PnL, les Frais et le Reward.
- **`tests/data/`** : Valider que les données ne contiennent pas de NaN ou d'incohérences après le téléchargement.
- **`tests/strategies/`** : Sanity check (run de 100 steps) pour vérifier que le modèle ne crash pas.

### 2.2 Gestion des Erreurs (Resilience)
- Implémenter un décorateur `@retry` sur les appels API externes (Yahoo/Alpaca/Macro).
- Fallback automatique : Si `MacroData` échoue, utiliser une valeur neutre ou la dernière valeur connue plutôt que de crasher.

---

## 🧠 3. Intelligence Financière (Algo)

### 3.1 Détection de Régime de Marché (HMM)
Intégrer un module **Hidden Markov Model (HMM)** ou un Clustering (K-Means) pour classifier le marché en temps réel :
- *Bull / Bear / Sideways / High Volatility*.
- Le Reward ou l'Architecture du modèle pourra s'adapter dynamiquement au régime détecté.

### 3.2 Gestion de Configuration (Hydra)
Remplacer `argparse` et `hardware.py` par **Hydra** (`config.yaml` hiérarchique).
- Permet de lancer des expériences complexes : `python train.py model=lstm data=crypto hardware=server_runpod`.

---

## 📅 Roadmap d'Implémentation

### Phase 9.0 : Fondation (Tests)
- [x] Mise en place de `pytest` et premiers tests unitaires (`tests/test_trading_env.py`).
- [ ] Migration vers Hydra pour la configuration (Reporté).

### Phase 9.1 : Scalabilité (Completed ✅)
- [x] Implémentation du `SharedMemoryLoader` (`core/shared_memory_manager.py`).
- [x] Benchmark RAM vs V8 (Gain x10 confirmé).

### Phase 9.2 : Moteur de Données (Completed ✅)
- [x] Migration du `FeatureEngineer` vers Polars (`core/features.py`).
- [x] Gain de performance (0.09s vs 5s).

### Phase 9.3 : Intelligence (Semaine 4)
- [ ] Module `MarketRegimeDetector`.
- [ ] Intégration dans l'observation space du RL.

---
*Document généré par l'Architecte Technique - Février 2026*
