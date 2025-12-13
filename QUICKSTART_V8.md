# 🚀 Ploutos V8 Oracle - Guide de Démarrage Rapide

## ⚡ Installation Express (5 min)

### 1. Synchroniser le Code

```bash
cd ~/ai-factory/tmp/project_ploutos
git pull origin feature/v7-predictive-models
```

### 2. Installer les Dépendances

```bash
# Activer environnement virtuel
source ~/ai-factory/venv/bin/activate

# Installer nouvelles dépendances V8
pip install lightgbm xgboost ta scikit-learn

# Ou installer tout
pip install -r requirements.txt
```

---

## 🎯 Entraînement Rapide (10-15 min)

### Option A : Mode Rapide (4 tickers, ~10 min)

```bash
python src/train/train_v8_all.py --quick
```

### Option B : Mode Complet (12 tickers, ~20 min)

```bash
python src/train/train_v8_all.py
```

**Résultat attendu** :
```
✅ ENTRAINEMENT TERMINÉ

lightgbm_intraday:
  Train Accuracy: 70.32%
  Test Accuracy:  67.45%

xgboost_weekly:
  Train Accuracy: 72.18%
  Test Accuracy:  68.91%
```

---

## 🧪 Tests (2 min)

```bash
# Tests complets
python tests/test_v8_oracle.py

# Tests rapides
python tests/test_v8_oracle.py --quick

# Tester un modèle spécifique
python tests/test_v8_oracle.py --model lightgbm
```

---

## 💻 Utilisation Python

### Prédiction Simple

```python
from src.models.v8_oracle_ensemble import V8OracleEnsemble

# Charger Oracle
oracle = V8OracleEnsemble()
oracle.load_models()

# Prédiction multi-horizon
result = oracle.predict_multi_horizon('NVDA')

print(f"Ticker: {result['ticker']}")

for horizon, pred in result['predictions'].items():
    print(f"\n{horizon.upper()}:")
    print(f"  Prédiction: {pred['prediction']}")
    print(f"  Confiance: {pred['confidence']:.2f}%")

if 'ensemble' in result:
    ens = result['ensemble']
    print(f"\nENSEMBLE:")
    print(f"  {ens['prediction']} - {ens['confidence']:.2f}%")
    print(f"  Agreement: {ens['agreement']}")
```

### Recommandation de Trading

```python
# Recommandation avec différentes tolérances au risque
for risk in ['low', 'medium', 'high']:
    rec = oracle.get_recommendation('MSFT', risk_tolerance=risk)
    
    print(f"\nRisk {risk.upper()}:")
    print(f"  Action: {rec['action']}")
    print(f"  Strength: {rec['strength']}")
    print(f"  Confiance: {rec['confidence']:.2f}%")
```

### Analyse Batch

```python
tickers = ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN']

batch = oracle.batch_predict(tickers)

print("\nRésumé:")
print(f"  Bullish: {batch['summary']['bullish']}")
print(f"  Bearish: {batch['summary']['bearish']}")
print(f"  High confidence: {batch['summary']['high_confidence_count']}")
```

---

## 🌐 Dashboard Web

### Démarrer le Dashboard

```bash
python web/app.py
```

Ouvrir dans le navigateur : **http://localhost:5000**

### Endpoints API Disponibles

#### V8 Oracle (Nouveaux)
```
GET /api/v8/predict/<ticker>     # Prédiction multi-horizon
GET /api/v8/recommend/<ticker>   # Recommandation BUY/SELL/HOLD
GET /api/v8/batch?tickers=...    # Analyse batch
GET /api/v8/heatmap?tickers=...  # Heatmap de confiance
```

#### Compatibilité V7
```
GET /api/v7/analysis?ticker=NVDA
GET /api/v7/enhanced/predict/<ticker>
```

### Exemple cURL

```bash
# Prédiction NVDA
curl http://localhost:5000/api/v8/predict/NVDA

# Recommandation MSFT (risk medium)
curl "http://localhost:5000/api/v8/recommend/MSFT?risk=medium"

# Batch analysis
curl "http://localhost:5000/api/v8/batch?tickers=NVDA,MSFT,AAPL"
```

---

## 📊 Utilisation avec le Bot de Trading

### Intégration dans `main.py`

```python
from src.models.v8_oracle_ensemble import V8OracleEnsemble

# Dans la classe TradingBot
class TradingBot:
    def __init__(self):
        # ... code existant ...
        
        # Ajouter V8 Oracle
        self.oracle = V8OracleEnsemble()
        self.oracle.load_models()
    
    def get_trading_signal(self, ticker: str) -> dict:
        """
        Obtenir signal de trading avec V8 Oracle
        """
        # Recommandation Oracle
        rec = self.oracle.get_recommendation(
            ticker, 
            risk_tolerance='medium'
        )
        
        # Combiner avec PPO ou autres stratégies
        if rec['action'] == 'BUY' and rec['confidence'] > 70:
            return {'action': 'BUY', 'confidence': rec['confidence']}
        elif rec['action'] == 'SELL' and rec['confidence'] > 70:
            return {'action': 'SELL', 'confidence': rec['confidence']}
        else:
            return {'action': 'HOLD', 'confidence': rec['confidence']}
```

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

### Modèles non chargés
```bash
# Vérifier que les modèles existent
ls -la models/

# Si vide, entraîner
python src/train/train_v8_all.py --quick
```

### Accuracy faible (<60%)
1. Ré-entraîner avec plus de données
2. Augmenter la période (2023-2024 minimum)
3. Ajouter plus de tickers diversifiés

---

## 📚 Documentation Complète

Pour plus de détails, consulter :
- **[Documentation V8 Oracle](docs/V8_ORACLE_README.md)**
- **Code source** : `src/models/v8_*.py`
- **Tests** : `tests/test_v8_oracle.py`

---

## ✅ Checklist de Vérification

- [ ] Dépendances installées (`lightgbm`, `xgboost`, `ta`)
- [ ] Modèles entraînés (fichiers `.pkl` dans `models/`)
- [ ] Tests passés (`python tests/test_v8_oracle.py`)
- [ ] Dashboard accessible (`http://localhost:5000`)
- [ ] API fonctionne (`curl http://localhost:5000/api/v8/predict/NVDA`)

---

## 🎉 Prochaines Étapes

1. **Entraîner** les modèles : `python src/train/train_v8_all.py`
2. **Tester** : `python tests/test_v8_oracle.py`
3. **Lancer dashboard** : `python web/app.py`
4. **Intégrer** dans le bot de trading
5. **Monitorer** les performances en production

---

**Happy Trading with V8 Oracle! 🚀📊**
