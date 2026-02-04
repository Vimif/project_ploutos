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

## Comment l'utiliser

### Entraîner le modèle

```bash
# Entraînement standard
python scripts/train_v4_optimal.py

# Avec une config personnalisée
python scripts/train_v4_optimal.py --config config/training_config_v6_better_timing.yaml
```

### Tester un modèle

```bash
python scripts/validate.py models/mon_model.zip
```

### Lancer le monitoring en production

```bash
# Surveiller les performances
python scripts/monitor_production.py --model models/mon_model.zip

# Avec ré-entraînement automatique si dérive détectée
python scripts/monitor_production.py --model models/mon_model.zip --auto-retrain
```

---

## Structure du projet

```
project_ploutos/
├── config/           # Fichiers de configuration YAML
├── core/             # Code principal
│   ├── data_fetcher.py       # Récupération des données
│   ├── features.py           # Calcul des indicateurs
│   ├── risk_manager.py       # Gestion du risque
│   └── universal_environment_v6_better_timing.py  # Environnement Gym
├── trading/          # Logique de trading live
├── training/         # Scripts d'entraînement
├── scripts/          # Points d'entrée CLI
└── docs/             # Documentation détaillée
```

---

## Configuration

Édite `config/autonomous_config.yaml` :

```yaml
training:
  timesteps: 2000000
  n_envs: 8
  device: "cuda"  # ou "cpu"

monitoring:
  sensitivity: "medium"  # low, medium, high
  auto_retrain: false
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
- [x] Détection de drift du modèle
- [x] Walk-forward validation

**En cours :**
- [ ] Détection des régimes de marché
- [ ] Ensemble de modèles
- [ ] Amélioration du système de récompense

**Futur :**
- [ ] Architecture Transformer
- [ ] Meta-learning (MAML)

---

## License

MIT

---

*Dernière mise à jour : Février 2026*
