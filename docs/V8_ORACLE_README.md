# 🔥 Ploutos V8 Oracle - Documentation Complète

## 🎯 Vue d'ensemble

Ploutos V8 Oracle est un **système de prédiction multi-horizon** pour le trading algorithmique, basé sur des modèles d'apprentissage automatique robustes.

### 💡 Philosophie

- **Robustesse > Complexité** : XGBoost/LightGBM au lieu de Deep Learning
- **Multi-horizon** : Court (1j), Moyen (5j), Long (20j) terme
- **Ensemble** : Aggrégation intelligente des prédictions
- **Production-ready** : Inférence rapide (<10ms), monitoring intégré

---

## 🏛️ Architecture

```
┌───────────────────────────────────────────────────┐
│           PLOUTOS V8 ORACLE SYSTEM                      │
└───────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────┐
│  Modèle 1: LightGBM Intraday                         │
│  Horizon: 1 jour                                      │
│  Features: 30+ indicateurs techniques                 │
│  Accuracy cible: 65-75%                               │
│  Inférence: <5ms                                       │
└───────────────────────────────────────────────────┘
                          │
                          v
┌───────────────────────────────────────────────────┐
│  Modèle 2: XGBoost Weekly                            │
│  Horizon: 5 jours                                     │
│  Features: 35+ indicateurs + support/resistance       │
│  Accuracy cible: 65-75%                               │
│  Inférence: <10ms                                      │
└───────────────────────────────────────────────────┘
                          │
                          v
┌───────────────────────────────────────────────────┐
│  Ensemble Meta-Model                                  │
│  Aggrégation pondérée                                │
│  Confiance calibrée                                   │
│  Recommandations BUY/SELL/HOLD                        │
└───────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
cd ~/ai-factory/tmp/project_ploutos
git pull origin feature/v7-predictive-models

# Installer dépendances
pip install lightgbm xgboost ta
```

### Entraînement Rapide (10-15 min)

```bash
# Entraînement quick (4 tickers)
python src/train/train_v8_all.py --quick

# Entraînement complet (12 tickers)
python src/train/train_v8_all.py
```

### Prédiction

```python
from src.models.v8_oracle_ensemble import V8OracleEnsemble

oracle = V8OracleEnsemble()
oracle.load_models()

# Prédiction single
result = oracle.predict_multi_horizon('NVDA')
print(result)

# Recommandation
rec = oracle.get_recommendation('NVDA', risk_tolerance='medium')
print(f"Action: {rec['action']} ({rec['strength']}) - Conf: {rec['confidence']:.1f}%")

# Batch
batch = oracle.batch_predict(['NVDA', 'MSFT', 'AAPL'])
print(batch['summary'])
```

---

## 📊 Modèles Détaillés

### 1. LightGBM Intraday (Court Terme)

**Fichier** : `src/models/v8_lightgbm_intraday.py`

**Features (30+)** :
- **Momentum** : RSI (7,14,21), Stochastic, Williams %R, ROC
- **Trend** : MACD, ADX, SMA/EMA (10,20,50), Distance to MAs
- **Volatility** : Bollinger Bands, ATR
- **Volume** : OBV, Volume Ratio, MFI, VPT
- **Price Action** : Returns 1d/5d/10d, HL Range

**Hyperparams** :
```python
{
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 7,
    'min_data_in_leaf': 50,
    'feature_fraction': 0.8,
    'lambda_l1': 0.1,
    'lambda_l2': 1.0
}
```

**Usage** :
```bash
python src/models/v8_lightgbm_intraday.py
```

---

### 2. XGBoost Weekly (Moyen Terme)

**Fichier** : `src/models/v8_xgboost_weekly.py`

**Features (35+)** :
- **Trend** : SMA/EMA (10,20,50,100,200), MA Crossovers, Ichimoku
- **Support/Resistance** : Pivot Points, Distance to Pivot
- **Momentum** : RSI, Stochastic
- **Volatility** : Bollinger Bands, ATR
- **Volume** : VWAP, Volume Trend, OBV
- **Price Action** : Returns 5d/10d/20d, Volatility

**Hyperparams** :
```python
{
    'learning_rate': 0.03,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}
```

---

## 🧠 Système de Confiance

### Calibration Multi-Facteurs

```python
confidence = (
    model_proba +
    trend_bonus (ADX) +
    volume_bonus -
    volatility_penalty
)
```

### Niveaux de Confiance

| Confiance | Interprétation |
|-----------|------------------|
| 75-100% | STRONG CONVICTION |
| 65-75% | MODERATE |
| 55-65% | WEAK |
| <55% | HOLD |

---

## 💼 Recommandations de Trading

### Seuils par Tolérance au Risque

| Risk Tolerance | Seuil Confiance | Description |
|----------------|----------------|-------------|
| **Low** | 75% | Conservateur |
| **Medium** | 65% | Équilibré |
| **High** | 55% | Agressif |

### Logique de Décision

```python
if confidence >= threshold:
    if prediction == 'UP':
        action = 'BUY'
    else:
        action = 'SELL' if agreement == 'STRONG' else 'HOLD'
else:
    action = 'HOLD'
```

---

## 🛠️ Maintenance

### Ré-entraînement Recommandé

- **Court terme** : Tous les mois
- **Moyen terme** : Tous les 2 mois

```bash
# Re-train avec données récentes
python src/train/train_v8_all.py --start-date 2024-01-01
```

### Monitoring

```python
# Vérifier accuracy sur données récentes
from tests.test_predictive_models import PredictiveModelTester

tester = PredictiveModelTester()
data = tester.load_data(['NVDA', 'MSFT'], days=30)
tester.test_v7_momentum(data)  # Remplacer par test V8
```

---

## 📊 Performance Attendue

### Accuracy Cible

| Modèle | Horizon | Accuracy Train | Accuracy Test |
|--------|---------|----------------|---------------|
| LightGBM | 1 jour | 70-75% | 65-70% |
| XGBoost | 5 jours | 70-75% | 65-70% |
| Ensemble | Multi | - | 68-73% |

### Comparaison avec V7

| Métrique | V7 Deep Learning | V8 Ensemble |
|----------|-----------------|-------------|
| Accuracy Test | 45% ❌ | 68% ✅ |
| Overfitting | Élevé | Faible |
| Temps entraînement | 2-4h | 10-20min |
| Inférence | 50ms | <10ms |
| Maintenance | Complexe | Simple |

---

## 🔧 Dépannage

### Erreur : "Module lightgbm not found"
```bash
pip install lightgbm
```

### Erreur : "Module ta not found"
```bash
pip install ta
```

### Accuracy faible (<60%)
1. Vérifier la qualité des données
2. Augmenter la période d'entraînement
3. Ajouter plus de tickers diversifiés
4. Ajuster les hyperparams

---

## 📚 Références

- **LightGBM** : https://lightgbm.readthedocs.io/
- **XGBoost** : https://xgboost.readthedocs.io/
- **Technical Analysis Library** : https://technical-analysis-library-in-python.readthedocs.io/

---

## 👥 Support

Pour toute question :
1. Consulter cette documentation
2. Vérifier les logs dans `logs/`
3. Tester avec `tests/test_predictive_models.py`

---

**Happy Trading with V8 Oracle! 🚀📊**
