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

### 1. Entraînement Walk-Forward

```bash
# PPO standard (recommandé pour commencer)
python training/train_walk_forward.py --config config/training_config_v8.yaml

# RecurrentPPO avec LSTM (capture les dépendances temporelles)
python training/train_walk_forward.py --config config/training_config_v8.yaml --recurrent

# Ensemble de 3 modèles (plus robuste, recommandé pour production)
python training/train_walk_forward.py --config config/training_config_v8.yaml --ensemble 3
```

### 2. Optimisation des hyperparamètres (optionnel)

```bash
python scripts/optimize_hyperparams.py --config config/training_config_v8.yaml --n-trials 50
```

### 3. Tests de robustesse

```bash
# Tous les tests
python scripts/robustness_tests.py --model models/v8/model.zip --all

# Monte Carlo (1000 simulations)
python scripts/robustness_tests.py --model models/v8/model.zip --monte-carlo 1000

# Stress test (crash de -30%)
python scripts/robustness_tests.py --model models/v8/model.zip --stress-test --crash-pct -0.30
```

### 4. Paper trading

```bash
python scripts/paper_trade_v7.py
```

> **GPU Cloud** : Pour un entraînement ~10x plus rapide, voir le [guide RunPod](docs/RUNPOD_GUIDE.md).

---

## Structure du projet

```
project_ploutos/
├── config/           # Configuration (YAML V8 + dataclasses)
├── core/             # Code principal
│   ├── universal_environment_v8_lstm.py  # Environnement Gym (V8)
│   ├── data_fetcher.py       # Récupération des données (Yahoo Finance)
│   ├── macro_data.py         # Indicateurs macro (VIX/TNX/DXY)
│   ├── ensemble.py           # Ensemble multi-modèles
│   ├── data_pipeline.py      # Feature engineering
│   └── risk_manager.py       # Gestion du risque
├── trading/          # Intégrations broker (eToro, Alpaca)
├── training/         # Walk-forward training (V8)
├── scripts/          # CLI (backtest, optimisation, paper trade)
└── docs/             # Documentation
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

**En cours :**
- [ ] Détection des régimes de marché
- [ ] Amélioration du système de récompense

**Futur :**
- [ ] Architecture Transformer
- [ ] Meta-learning (MAML)

---

## License

MIT

---

*Dernière mise à jour : Février 2026*
