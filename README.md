# Ploutos Trading V9 (Polars + Shared Memory)

Un projet personnel de trading algorithmique utilisant le Reinforcement Learning. L'idée : entraîner un agent à trader de manière autonome sur les marchés financiers.

> ⚠️ **Avertissement** : Ce projet est expérimental et en paper trading. Le trading algorithmique comporte des risques significatifs. Ne jamais utiliser d'argent réel sans comprendre ces risques.

---

## C'est quoi ?

Ploutos est un bot de trading qui apprend par lui-même en utilisant l'algorithme PPO (Proximal Policy Optimization). Au lieu de suivre des règles fixes, il observe le marché et développe sa propre stratégie.

**Ce que ça fait :**
- Collecte les données de marché (via Alpaca)
- Analyse les tendances avec 85+ indicateurs techniques (Moteur Polars ultra-rapide)
- Prend des décisions d'achat/vente de manière autonome
- Utilise la Shared Memory pour un entraînement parallèle sans surcharger la RAM
- Se ré-entraîne automatiquement si nécessaire

---

## Performances actuelles

| Métrique | Valeur |
|----------|--------|
| Sharpe Ratio | ~1.5 |
| Max Drawdown | -12% |
| Win Rate | 55% |
| **Speed (Features)** | **x100 (0.09s/100k bars)** |
| Mode | Paper Trading |

*Ces résultats sont en paper trading et ne garantissent rien en conditions réelles.*

---

## Installation

```bash
# Cloner le repo
git clone https://github.com/Vimif/project_ploutos
cd project_ploutos

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Installer les dépendances (Polars, PyArrow, Torch...)
pip install -e .
```

---

## Workflow d'entraînement (V9)

Le pipeline V9 utilise le walk-forward training avec support natif pour **Polars** et **Shared Memory**.

### 1. Pipeline High-Performance (Recommandé)

```bash
# Script optimisé pour le hardware (thread pinning, limit open files, auto-scale)
./start_training.sh
```

Ce script configure automatiquement l'environnement (OMP_NUM_THREADS), détecte le hardware (GPU/RAM) et lance le pipeline complet avec les paramètres optimaux.

### 2. Entraînement Walk-Forward (séparé)

```bash
# PPO standard avec Shared Memory (V9)
python training/train.py --config config/config.yaml --auto-scale --shared-memory

# RecurrentPPO avec LSTM
python training/train.py --config config/config.yaml --recurrent --auto-scale --shared-memory

# Ensemble de 3 modèles
python training/train.py --config config/config.yaml --ensemble 3 --auto-scale --shared-memory
```

### 3. Optimisation des hyperparamètres (optionnel)

```bash
# Auto-détecte le nombre de jobs parallèles et n_envs par trial
python scripts/optimize_hyperparams.py --config config/config.yaml --n-trials 50 --auto-scale

# Ou manuellement : 4 trials parallèles
python scripts/optimize_hyperparams.py --config config/config.yaml --n-trials 50 --n-jobs 4
```

### 4. Tests de robustesse (séparé)

```bash
# Monte Carlo parallélisé + stress test
python scripts/robustness_tests.py --model models/<fold>/model.zip --vecnorm models/<fold>/vecnormalize.pkl --all --auto-scale

# Monte Carlo seul
python scripts/robustness_tests.py --model models/<fold>/model.zip --monte-carlo 1000 --auto-scale
```

### 5. Paper trading

```bash
# Lance le paper trading (détecte auto V9)
python scripts/paper_trade.py --model models/.../model.zip
```

> **GPU Cloud** : Avec `--auto-scale`, un seul config suffit pour dev et cloud. Voir le [guide RunPod](docs/RUNPOD_GUIDE.md).

---

## Structure du projet

```
project_ploutos/
├── config/             # Configuration
│   ├── hardware.py          # Auto-détection GPU/CPU/RAM + scaling
│   └── config.yaml          # Config training standard
├── core/               # Code principal V9
│   ├── environment.py       # Environnement V9 (Unified + SharedMem)
│   ├── features.py          # Moteur Polars (x100 speed)
│   ├── shared_memory_manager.py # Gestionnaire Shared Memory
│   ├── data_fetcher.py      # Récupération des données
│   └── risk_manager.py      # Gestion du risque
├── trading/            # Intégrations broker (eToro, Alpaca)
├── training/           # Module d'entraînement
│   └── train.py             # Script Walk-Forward V9
├── scripts/            # CLI (pipeline, optimisation, robustness, paper trade)
│   ├── run_pipeline.py      # Pipeline complet training→robustness
│   ├── paper_trade.py       # Paper Trading V9
│   └── ...
├── legacy/             # Archives (V6/V7/V8)
└── docs/               # Documentation
    ├── ARCHITECTURE_V9.md   # 🏗️ Architecture Technique V9
    ├── RELEASE_NOTES_V9.md  # 🚀 Nouveautés V9
    ├── RUNPOD_GUIDE.md      # Guide déploiement Cloud
    └── ...
```

---

## Configuration

Édite `config/config.yaml` :

```yaml
training:
  total_timesteps: 10000000  # par fold walk-forward
  n_envs: 16                 # Auto-scalé si --auto-scale
  use_shared_memory: true    # Activer V9 Shared Memory

walk_forward:
  train_years: 1       # Durée du training par fold
  test_months: 6       # Durée du test
  step_months: 6       # Pas entre chaque fold

wandb:
  enabled: false       # Activer pour le tracking
```

---

## Monitoring

**Logs** : `logs/train.log`

**Dashboards disponibles** :
- TensorBoard : `tensorboard --logdir models/walk_forward_.../`
- Grafana : `http://localhost:3000` (si configuré)

---

## La roadmap

**Fait :**
- [x] Curriculum Learning (apprentissage progressif)
- [x] Coûts de transaction réalistes
- [x] Walk-forward validation (V9)
- [x] Ensemble de modèles
- [x] Données macro (VIX/TNX/DXY)
- [x] RecurrentPPO (LSTM)
- [x] Déploiement cloud (RunPod)
- [x] Auto-scaling hardware (GPU/CPU/RAM)
- [x] **"Turbo Init" (Polars Engine x100)**
- [x] **Optimisation RAM (Shared Memory)**
- [x] Protection contre le Data Leakage (Embargo)
- [x] Tests de Robustesse (Monte Carlo + PSR/DSR)

**Prochaines Étapes :**
- [ ] Tests Unitaires & CI/CD (Pytest 100% coverage)
- [ ] Détection des régimes de marché (HMM/Clustering)

**Futur :**
- [ ] Architecture Transformer
- [ ] Meta-learning (MAML)

---

## License

MIT

---

*Dernière mise à jour : Février 2026 (V9)*
