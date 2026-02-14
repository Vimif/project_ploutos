# Ploutos Trading

Un projet personnel de trading algorithmique utilisant le Reinforcement Learning. L'idée : entraîner un agent à trader de manière autonome sur les marchés financiers.

> ⚠️ **Avertissement** : Ce projet est expérimental et en paper trading. Le trading algorithmique comporte des risques significatifs. Ne jamais utiliser d'argent réel sans comprendre ces risques.

---

## C'est quoi ?

Ploutos est un bot de trading qui apprend par lui-même en utilisant l'algorithme PPO (Proximal Policy Optimization). Au lieu de suivre des règles fixes, il observe le marché et développe sa propre stratégie.

**Ce que ça fait :**
- Collecte les données de marché (via Alpaca)
- Analyse les tendances avec des indicateurs techniques
- Prend des décisions d'achat/vente de manière autonome
- Détecte quand ses performances se dégradent (drift detection)
- Se ré-entraîne automatiquement si nécessaire

---

## Performances actuelles

| Métrique | Valeur |
|----------|--------|
| Sharpe Ratio | ~1.5 |
| Max Drawdown | -12% |
| Win Rate | 55% |
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

# Installer les dépendances
pip install -e .
```

---

## Workflow d'entraînement (V8)

Le pipeline optimisé utilise le walk-forward training avec validation glissante.

### 1. Pipeline High-Performance (Recommandé)

```bash
# Script optimisé pour le hardware (thread pinning, limit open files, auto-scale)
./start_training.sh
```

Ce script configure automatiquement l'environnement (OMP_NUM_THREADS), détecte le hardware (GPU/RAM) et lance le pipeline complet avec les paramètres optimaux (Batch 65k, 256 Envs si possible).

### 2. Entraînement Walk-Forward (séparé)

```bash
# PPO standard
python training/train_walk_forward.py --config config/training_config_v8.yaml --auto-scale

# RecurrentPPO avec LSTM
python training/train_walk_forward.py --config config/training_config_v8.yaml --recurrent --auto-scale

# Ensemble de 3 modèles
python training/train_walk_forward.py --config config/training_config_v8.yaml --ensemble 3 --auto-scale
```

### 3. Optimisation des hyperparamètres (optionnel)

```bash
# Auto-détecte le nombre de jobs parallèles et n_envs par trial
python scripts/optimize_hyperparams.py --config config/training_config_v8.yaml --n-trials 50 --auto-scale

# Ou manuellement : 4 trials parallèles
python scripts/optimize_hyperparams.py --config config/training_config_v8.yaml --n-trials 50 --n-jobs 4
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
python scripts/paper_trade_v7.py
```

> **GPU Cloud** : Avec `--auto-scale`, un seul config suffit pour dev et cloud. Voir le [guide RunPod](docs/RUNPOD_GUIDE.md).

---

## Structure du projet

```
project_ploutos/
├── config/           # Configuration
│   ├── hardware.py            # Auto-détection GPU/CPU/RAM + scaling
│   ├── settings.py            # Chemins, broker, WandB
│   ├── training_config_v8.yaml         # Config training standard
│   └── training_config_v8_cloud.yaml   # Config cloud (override manuel)
├── core/             # Code principal
│   ├── universal_environment_v8_lstm.py  # Environnement Gym (V8)
│   ├── data_fetcher.py       # Récupération des données (Yahoo Finance)
│   ├── macro_data.py         # Indicateurs macro (VIX/TNX/DXY)
│   ├── ensemble.py           # Ensemble multi-modèles
│   ├── data_pipeline.py      # Feature engineering
│   └── risk_manager.py       # Gestion du risque
├── trading/          # Intégrations broker (eToro, Alpaca)
├── training/         # Walk-forward training (V8)
├── scripts/          # CLI (pipeline, optimisation, robustness, paper trade)
│   ├── run_pipeline.py       # Pipeline complet training→robustness
│   ├── optimize_hyperparams.py  # Optuna hyperparameter search
│   └── robustness_tests.py   # Monte Carlo + stress tests
└── docs/             # Documentation
    ├── AUDIT_TECHNIQUE_V8.md # 🛡️ Audit Technique & Architecture (Fev 2026)
    ├── DEV_KNOWLEDGE.md      # Base de connaissance développeur
    └── RUNPOD_GUIDE.md       # Guide déploiement Cloud
```

---

## Configuration

Édite `config/training_config_v8.yaml` :

```yaml
training:
  total_timesteps: 10000000  # par fold walk-forward
  n_envs: 8
  learning_rate: 0.0001

walk_forward:
  train_years: 1       # Durée du training par fold
  test_months: 6       # Durée du test
  step_months: 6       # Pas entre chaque fold

wandb:
  enabled: false       # Activer pour le tracking
```

---

## Monitoring

**Logs** : `logs/ploutos_*.log`

**Dashboards disponibles** :
- TensorBoard : `tensorboard --logdir logs/tensorboard`
- Grafana : `http://localhost:3000` (si configuré)

---

## La roadmap

**Fait :**
- [x] Curriculum Learning (apprentissage progressif)
- [x] Coûts de transaction réalistes
- [x] Walk-forward validation (V8)
- [x] Ensemble de modèles
- [x] Données macro (VIX/TNX/DXY)
- [x] RecurrentPPO (LSTM)
- [x] Déploiement cloud (RunPod)
- [x] Auto-scaling hardware (GPU/CPU/RAM)
- [x] Pipeline orchestrateur (training + robustness)
- [x] "Turbo Init" (Pre-computed Features)
- [x] Amélioration du système de récompense (Differential Sharpe Ratio)
- [x] Protection contre le Data Leakage (Embargo)
- [x] Tests de Robustesse (Monte Carlo + PSR/DSR)

**Prochaines Étapes (V9) :**
- [ ] Tests Unitaires & CI/CD (Pytest)
- [ ] Optimisation RAM (Shared Memory)
- [ ] Détection des régimes de marché (HMM/Clustering)

**Futur :**
- [ ] Architecture Transformer
- [ ] Meta-learning (MAML)

---

## License

MIT

---

*Dernière mise à jour : Février 2026*
