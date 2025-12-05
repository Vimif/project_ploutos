# 🤖 Ploutos Trading v2.0

Système de trading algorithmique autonome avec Reinforcement Learning (PPO).

## ✨ Nouveautés v2.0

- 🏗️ **Architecture refactorisée** : Code modulaire et maintenable
- 📊 **Logger centralisé** : Logs structurés fichier + console
- ⚙️ **Configuration unifiée** : YAML + dataclasses typées
- 🧪 **Tests unitaires** : Couverture 60%+
- 🚀 **Scripts simplifiés** : CLI claire et intuitive

## 📦 Installation

Cloner

git clone https://github.com/Vimif/project_ploutos
cd project_ploutos
Virtualenv

python3 -m venv venv
source venv/bin/activate
Dépendances

pip install -e .

text

## 🚀 Usage

### Entraînement

Simple

python3 scripts/train.py
Custom config

python3 scripts/train.py --config config/my_config.yaml
Output spécifique

python3 scripts/train.py --output models/my_model.zip

text

### Validation

Valider un modèle

python3 scripts/validate.py models/autonomous/trained_model.zip

text

### Déploiement

Déployer en production

python3 scripts/deploy.py models/autonomous/trained_model.zip

text

## 📁 Structure

project_ploutos/
├── config/ # Configuration
├── core/ # Modules principaux
│ ├── agents/ # Trainer, Validator, Deployer
│ ├── data/ # Data fetching
│ ├── environments/# Gym environments
│ └── market/ # Regime detection, asset selection
├── utils/ # Utilitaires
├── scripts/ # Points d'entrée
└── tests/ # Tests unitaires

text

## 🧪 Tests

Lancer tous les tests

pytest
Avec couverture

pytest --cov
Test spécifique

pytest tests/test_config.py

text

## 📊 Monitoring

- **Logs** : `logs/ploutos_YYYYMMDD_HHMMSS.log`
- **TensorBoard** : `tensorboard --logdir logs/tensorboard`
- **W&B** : Configure dans script

## 🔧 Configuration

Éditer `config/autonomous_config.yaml`:

training:
timesteps: 2000000
n_envs: 8
device: "cuda"
learning_rate: 0.0001

text

## 📝 License

MIT