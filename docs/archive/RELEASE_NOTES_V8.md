# Ploutos V8.1 Release Notes - Février 2026

## 🚀 Nouvelles Fonctionnalités

### 1. Architecture "Turbo" & Hardware Scaling
- **Parallelisme Massif** : Support natif pour 256+ environnements simultanés.
- **Start-up Instantané** : "Turbo Init" pré-calcule 100% des indicateurs techniques avant de spawner les processus, réduisant le temps de démarrage de 10 min à <10s.
- **Auto-Scale** : Détection du hardware (CPU/GPU/RAM) pour ajuster `n_envs` et `batch_size`.
- **RAM Protection** : Cap automatique à 128 envs si RAM < 128GB pour éviter les OOM.

### 2. Robustesse Financière (Quantitative)
- **Differential Sharpe Ratio (DSR)** : L'IA optimise désormais le ratio de Sharpe glissant plutôt que le PnL brut, favorisant la régularité.
- **Embargo (Anti-Leak)** : Implementation formelle d'un "buffer" de 1 mois entre Train et Test dans le Walk-Forward pour empêcher les indicateurs (RSI, EMA) de "voir le futur".
- **Macro Data integration** : Le réseau reçoit désormais VIX, TNX et DXY en entrée directe.

### 3. Modèle & Entraînement
- **RecurrentPPO (LSTM)** : Support complet des réseaux récurrents pour la mémoire temporelle.
- **Ensemble Learning** : Pipeline natif pour entraîner N modèles en parallèle et moyenner leurs prédictions.
- **Penalized Reward** : Pénalités dynamiques pour l'overtrading (Turnover) et le Drawdown.

---

## 🛠️ Changements Techniques (Breaking Changes)

### Dépendances
- Ajout de `sb3-contrib` (pour RecurrentPPO).
- Nécessite `pandas >= 2.0`.

### Scripts
- **Nouveau Standard** : `./start_training.sh` est le point d'entrée unique recommandé pour l'entraînement. Il gère les variables d'environnement critiques (`OMP_NUM_THREADS=1`) pour éviter le "Thread Explosion".
- `scripts/build_dataset.py` : Utilise désormais un cache local dans `data/dataset_v8/`.

---

## 📊 Performance Attendue

- **Sharpe Ratio** : > 1.5 sur Test OOS.
- **Vitesse d'entraînement** : ~5000 FPS sur machine standard, ~50 000+ FPS sur serveur High-End.
- **Stabilité** : Converge sans "Collapse" grâce au DSR.
