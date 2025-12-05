# 🤖 Ploutos Trading v2.0

Système de trading algorithmique autonome avec Reinforcement Learning (PPO).

## ✨ Nouveautés v2.0

- 🏭 **Architecture refactorisée** : Code modulaire et maintenable
- 📊 **Logger centralisé** : Logs structurés fichier + console
- ⚙️ **Configuration unifiée** : YAML + dataclasses typées
- 🧪 **Tests unitaires** : Couverture 60%+
- 🚀 **Scripts simplifiés** : CLI claire et intuitive
- **🔍 Model Drift Detection** : Détection automatique dérive (PSI, KS Test, ADDM) 🆕

---

## 📊 Performances

| Métrique | Valeur |
|----------|--------|
| **Sharpe Ratio** | 1.5+ |
| **Max Drawdown** | -12% |
| **Win Rate** | 55% |
| **Environnement** | Paper Trading |

---

## 📚 Documentation

- [Architecture Détaillée](docs/ARCHITECTURE.md)
- **[Guide Monitoring Production](docs/MONITORING.md)** 🆕 **NOUVEAU**
- [Configuration Bot](docs/BOT_CONFIG.md)

---

## 📎 Quick Links

- **Monitoring Dashboard** : `http://localhost:3000` (Grafana)
- **Prometheus** : `http://localhost:9090`
- **Weights & Biases** : [Ploutos Project](https://wandb.ai)

---

## 📦 Installation

### Cloner

```bash
git clone https://github.com/Vimif/project_ploutos
cd project_ploutos
```

### Virtualenv

```bash
python3 -m venv venv
source venv/bin/activate
```

### Dépendances

```bash
pip install -e .
```

---

## 🚀 Usage

### Entraînement

```bash
# Simple
python3 scripts/train.py

# Custom config
python3 scripts/train.py --config config/my_config.yaml

# Output spécifique
python3 scripts/train.py --output models/my_model.zip

# Curriculum Learning (recommandé)
python3 scripts/train_curriculum.py --stage 1
python3 scripts/train_curriculum.py --stage 2 --load-model models/stage1_final
```

---

### Validation

```bash
# Valider un modèle
python3 scripts/validate.py models/autonomous/trained_model.zip
```

---

### Monitoring Production 🆕 **NOUVEAU**

```bash
# Monitoring simple
python3 scripts/monitor_production.py --model models/stage1_final.zip

# Avec auto-retrain si dérive
python3 scripts/monitor_production.py --model models/stage1_final.zip --auto-retrain

# Haute sensibilité (détection agressive)
python3 scripts/monitor_production.py --model models/stage1_final.zip --sensitivity high
```

**Détecte 3 types de dérive** :
- **Data Drift** : Distribution features change (PSI + KS Test)
- **Concept Drift** : Relation X→Y change (ADDM)
- **Model Drift** : Performance se dégrade

📚 **[Documentation complète](docs/MONITORING.md)**

---

### Déploiement

```bash
# Déployer en production
python3 scripts/deploy.py models/autonomous/trained_model.zip
```

---

## 📁 Structure

```
project_ploutos/
├── config/               # Configuration
├── core/                 # Modules principaux
│   ├── agents/          # Trainer, Validator, Deployer
│   ├── data/            # Data fetching
│   ├── environments/    # Gym environments
│   ├── market/          # Regime detection, asset selection
│   └── drift_detector.py # 🆕 Détection dérive
├── utils/                # Utilitaires
├── scripts/              # Points d'entrée
│   ├── train_curriculum.py
│   └── monitor_production.py # 🆕 Monitoring
├── docs/                 # Documentation
│   └── MONITORING.md     # 🆕 Guide monitoring
└── tests/                # Tests unitaires
```

---

## 🧪 Tests

```bash
# Lancer tous les tests
pytest

# Avec couverture
pytest --cov

# Test spécifique
pytest tests/test_config.py

# Test drift detector
python3 core/drift_detector.py
```

---

## 📊 Monitoring

### **Logs**
- Application : `logs/ploutos_YYYYMMDD_HHMMSS.log`
- Drift Events : `logs/drift_events.jsonl`

### **Dashboards**
- **TensorBoard** : `tensorboard --logdir logs/tensorboard`
- **Grafana** : `http://localhost:3000` (VPS uniquement)
- **Prometheus** : `http://localhost:9090`

### **Tracking**
- **Weights & Biases** : Configure dans script
- **Drift Reports** : `reports/drift_monitoring_latest.json`

---

## 🔧 Configuration

Éditer `config/autonomous_config.yaml`:

```yaml
training:
  timesteps: 2000000
  n_envs: 8
  device: "cuda"
  learning_rate: 0.0001

monitoring:
  sensitivity: "medium"  # low|medium|high
  auto_retrain: false
  check_frequency: "daily"  # hourly|daily|weekly
```

---

## ✨ Fonctionnalités Principales

### **1. Curriculum Learning**
- Stage 1 : Mono-Asset (SPY)
- Stage 2 : Multi-Asset ETFs
- Stage 3 : Actions complexes

### **2. Coûts Réalistes**
- Commissions + Slippage + Spread
- Impact de marché

### **3. Walk-Forward Validation**
- Validation temporelle
- Évite overfitting

### **4. Model Drift Detection** 🆕
- PSI (Population Stability Index)
- KS Test (Kolmogorov-Smirnov)
- ADDM (Autoregressive Drift Detection)
- Auto-Retrain optionnel

---

## 🛡️ Sécurité

- Max position size : 50% capital
- Stop-loss dynamique
- Drawdown limit : -20%
- Monitoring 24/7
- **Drift detection** : Alertes automatiques

---

## ⚠️ Avertissements

- 🚨 **Paper Trading** : Système actuellement en paper trading
- ⚠️ **Risques** : Trading algorithmique comporte des risques
- 🔍 **Monitoring** : Surveillance quotidienne recommandée

---

## 📈 Roadmap

### **Phase 1** ✅ (Complétée)
- [x] Curriculum Learning
- [x] Coûts réalistes
- [x] Walk-Forward Validation
- [x] Model Drift Detection

### **Phase 2** 🔄 (En cours)
- [ ] Ensemble Models
- [ ] Market Regime Detection
- [ ] Advanced Reward Shaping

### **Phase 3** 🔮 (Futur)
- [ ] Adversarial Training
- [ ] Meta-Learning (MAML)
- [ ] Transformer Architecture

---

## 📝 License

MIT

---

**Dernière mise à jour** : 5 décembre 2025
