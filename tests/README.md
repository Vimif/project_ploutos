# 🧪 Ploutos Testing Framework

Système de test complet pour évaluer et comparer les modèles de trading sans deployer en production.

## 🎯 Objectifs

- ✅ **Backtesting rapide** sur données historiques
- ✅ **Comparaison A/B** entre modèles (PPO vs V7 vs Hybrid)
- ✅ **Métriques détaillées** (return, Sharpe, drawdown, win rate)
- ✅ **Tests reproductibles** avec configuration flexible
- ✅ **Rapports JSON** pour analyse ultérieure

---

## 🚀 Quick Start

### Test rapide avec preset

```bash
cd ~/ploutos/project_ploutos

# Test sur actions tech (90 derniers jours)
python tests/quick_test.py --preset tech

# Test sur actions financières (30 jours)
python tests/quick_test.py --preset finance --days 30

# Test complet (10 tickers, 180 jours)
python tests/quick_test.py --preset full --days 180
```

### Test personnalisé

```bash
# Tickers spécifiques
python tests/quick_test.py --tickers NVDA,MSFT,AAPL,SPY

# Capital initial personnalisé
python tests/quick_test.py --preset mixed --capital 50000

# Modèle personnalisé
python tests/quick_test.py --model models/my_model.zip --preset tech
```

---

## 📊 Métriques Calculées

| Métrique | Description |
|----------|-------------|
| **Total Return (%)** | Performance totale du portfolio |
| **Final Value ($)** | Valeur finale du portfolio |
| **Total Trades** | Nombre de transactions exécutées |
| **Win Rate (%)** | Pourcentage de trades gagnants |
| **Sharpe Ratio** | Ratio risque/rendement (annualisé) |
| **Max Drawdown (%)** | Perte maximale depuis le pic |

---

## 🔧 Presets Disponibles

### `tech` - Actions technologiques
```python
['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'TSLA']
```

### `finance` - Secteur financier
```python
['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK']
```

### `energy` - Secteur énergétique
```python
['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC']
```

### `defensive` - ETFs défensifs
```python
['SPY', 'QQQ', 'VOO', 'VTI', 'IWM', 'DIA', 'VEA']
```

### `mixed` - Mix diversifié (défaut)
```python
['NVDA', 'MSFT', 'JPM', 'XOM', 'SPY', 'QQQ', 'AAPL']
```

### `full` - Test complet
```python
['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'JPM', 'SPY', 'QQQ', 'XOM', 'CVX']
```

---

## 📝 Options Avancées

```bash
# Tester seulement PPO (skip V7)
python tests/quick_test.py --preset tech --skip-v7 --skip-hybrid

# Changer le split train/test (70/30 par défaut)
python tests/quick_test.py --preset mixed --test-split 0.2

# Ajuster les frais de transaction
python tests/quick_test.py --preset finance --commission 0.002
```

---

## 📁 Utilisation Avancée - Framework Python

Pour des tests personnalisés complexes :

```python
from tests.backtest_framework import BacktestFramework

# Initialiser
framework = BacktestFramework(
    initial_capital=100000,
    commission=0.001,
    slippage=0.0005
)

# Charger données
data = framework.load_historical_data(
    tickers=['NVDA', 'MSFT', 'AAPL'],
    start_date='2023-01-01',
    end_date='2024-12-01'
)

# Backtest PPO
metrics_ppo = framework.backtest_ppo_only(
    model_path='models/autonomous/production.zip',
    data=data,
    test_split=0.3
)

# Backtest V7
metrics_v7 = framework.backtest_v7_only(
    data=data,
    test_split=0.3
)

# Backtest Hybrid
metrics_hybrid = framework.backtest_ppo_plus_v7(
    model_path='models/autonomous/production.zip',
    data=data,
    test_split=0.3
)

# Comparer
comparison_df = framework.compare_models()
print(comparison_df)

# Sauvegarder
framework.save_results('tests/backtest_results')
```

---

## 📊 Exemple de Résultat

```
🏆 COMPARAISON DES MODÈLES
======================================================================

         Model  Return (%)  Final Value ($)  Trades  Win Rate (%)  Sharpe  Max DD (%)
    Ppo Only          8.45       108450.00      42          52.38    1.23       12.45
    V7 Only          12.34       112340.00      28          64.29    1.87        8.90
Ppo Plus V7          15.67       115670.00      35          65.71    2.14        7.23

🥇 Meilleur modèle: Ppo Plus V7
   Return: 15.67%
   Sharpe: 2.14
```

---

## ⚠️ Limitations

- **Données Yahoo Finance** : limitées à 730 jours pour données horaires
- **Slippage simulé** : 0.05% par défaut (peut varier en réalité)
- **Pas de coûts d'emprunt** : short selling non implémenté
- **Exécution parfaite** : pas de rejet d'ordre

---

## 💾 Structure des Résultats

Les résultats sont sauvegardés dans :

```
tests/backtest_results/
├── backtest_20251213_221530.json
├── backtest_20251213_223045.json
└── ...
```

Format JSON :

```json
{
  "ppo_only": {
    "metrics": {
      "total_return": 8.45,
      "total_trades": 42,
      "win_rate": 52.38,
      "sharpe_ratio": 1.23,
      "max_drawdown": 12.45,
      "final_value": 108450.0
    },
    "trade_count": 42
  },
  "v7_only": {...},
  "ppo_plus_v7": {...}
}
```

---

## 🔥 Conseils d'Utilisation

### 1. **Tests rapides en développement**
```bash
# 30 jours, 3-5 tickers
python tests/quick_test.py --preset mixed --days 30
```

### 2. **Validation avant production**
```bash
# 180 jours, 10 tickers, capital réel
python tests/quick_test.py --preset full --days 180 --capital 100000
```

### 3. **Tests spécifiques par secteur**
```bash
# Tester performance sur le secteur tech
python tests/quick_test.py --preset tech --days 90

# Comparer avec secteur finance
python tests/quick_test.py --preset finance --days 90
```

### 4. **Comparer plusieurs modèles PPO**
```bash
# Modèle actuel
python tests/quick_test.py --model models/autonomous/production.zip

# Nouveau modèle
python tests/quick_test.py --model models/new_model.zip
```

---

## 🛠️ Dépannage

### Erreur : "Pas assez de données"
```bash
# Augmenter la période
python tests/quick_test.py --days 180
```

### Erreur : "V7 non disponible"
```bash
# Vérifier que le modèle est bien chargé
ls -la models/v7_momentum_enhanced_best.pth

# Skip V7 si nécessaire
python tests/quick_test.py --skip-v7
```

### Erreur : "Modèle PPO introuvable"
```bash
# Vérifier le chemin
ls -la models/autonomous/production.zip

# Spécifier le bon chemin
python tests/quick_test.py --model path/to/your/model.zip
```

---

## 📚 Références

- **Sharpe Ratio** : https://en.wikipedia.org/wiki/Sharpe_ratio
- **Drawdown** : https://en.wikipedia.org/wiki/Drawdown_(economics)
- **Backtesting Best Practices** : https://www.investopedia.com/articles/trading/05/030205.asp

---

## 💬 Support

Pour toute question ou problème :
1. Vérifie la documentation ci-dessus
2. Consulte les logs dans `tests/backtest_results/`
3. Ouvre une issue sur GitHub

---

**Happy Testing! 🚀📊**
