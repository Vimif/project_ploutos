# 🏋️ Guide d'Entraînement V3 ULTIMATE - Ploutos Trading IA

## 🎯 Objectif

Ce guide décrit comment entraîner le modèle Ploutos V3 avec toutes les optimisations avancées pour maximiser les performances de trading.

## 🚀 Nouveautés V3

### Environnement V4 Ultra-Réaliste

- **Slippage dynamique** : Basé sur l'ATR (Average True Range)
- **Spread bid/ask** : 2 bps par défaut, configurable
- **Frais réels Alpaca** :
  - SEC fee: 0.00221%
  - FINRA TAF: 0.0145%
  - Commission: 0% (Alpaca commission-free)
- **Market impact** : Modélisation de l'impact sur le prix
- **PDT rules** : Pattern Day Trading (max 4 trades/jour)
- **Holding period** : Période de détention minimale

### 50+ Features Techniques

#### Trend Indicators (10)
- SMA 10/20/50
- EMA 10/20
- MACD (+ signal + diff)
- ADX (trend strength)

#### Momentum Indicators (12)
- RSI 7/14
- Stochastic (K, D)
- ROC 10/20
- Williams %R
- TSI, Ultimate Oscillator
- CCI, MFI

#### Volatility Indicators (8)
- Bollinger Bands (high, low, width, %)
- ATR 14
- Keltner Channel
- Ulcer Index

#### Volume Indicators (8)
- OBV, CMF, Force Index
- VPT, EOM
- Volume Ratio, VWAP

#### Support/Resistance (6)
- Pivot Points (R1, R2, S1, S2)
- 52-week high/low

#### Statistical Features (8)
- Rolling mean/std/skew/kurtosis
- Z-Score
- Autocorrelation
- Hurst exponent
- Variance ratio

#### Price Action (6)
- Body size
- Upper/Lower shadows
- Gap
- True Range
- Daily range

### Récompenses Sophistiquées

1. **Rendement de base** : `(equity_t - equity_t-1) / equity_t-1 * 100`
2. **Penalty Drawdown** : Si DD > 10%, pénalité = `DD * 10`
3. **Penalty Sharpe** : Si Sharpe < 0, pénalité = `Sharpe * 0.1`
4. **Penalty Overtrading** : `0.01 * n_trades` si > 1 trade
5. **Bonus Performance** : +0.5 si rendement > 2%

### Architecture Neuronale Profonde

```
Input (observations) 
  ↓
Dense(1024) + Tanh
  ↓
Dense(512) + Tanh
  ↓
Dense(256) + Tanh
  ↓
  ├─ Policy Head (actions)
  └─ Value Head (value function)
```

**Total params : ~2.5M**

### Hyperparams Optimisés

| Param | Valeur | Description |
|-------|--------|-------------|
| Learning Rate | 5e-5 | Très stable |
| Batch Size | 8192 | Large batch = stabilité |
| N Steps | 4096 | Steps avant update |
| N Epochs | 20 | Epochs par update |
| Gamma | 0.995 | Long terme |
| GAE Lambda | 0.98 | Variance reduction |
| Clip Range | 0.2 | PPO clipping |
| Entropy Coef | 0.01 | Exploration |
| Target KL | 0.015 | Adaptive KL |

## 💻 Prérequis

### Hardware

- **GPU recommandé** : RTX 3080 ou supérieur
- **RAM** : 16 GB minimum
- **VRAM** : 10 GB minimum
- **Stockage** : 50 GB disponibles

### Software

- Ubuntu 20.04+
- Python 3.8+
- CUDA 11.8+
- PyTorch 2.0+

## 🛠️ Installation

### Sur BBC (Machine GPU)

```bash
# 1. Aller dans le projet
cd /root/ai-factory/tmp/project_ploutos

# 2. Activer virtualenv
source /root/ai-factory/venv/bin/activate

# 3. Installer dépendances
bash scripts/install_training_deps.sh

# 4. Vérifier GPU
nvidia-smi
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## ⚡ Lancement Rapide

### Option 1 : Script Automatisé (Recommandé)

```bash
# Lancement interactif
bash scripts/launch_training_ultimate.sh

# Lancement en arrière-plan
bash scripts/launch_training_ultimate.sh --nohup

# Avec config custom
bash scripts/launch_training_ultimate.sh --config my_config.yaml
```

### Option 2 : Python Direct

```bash
# Config par défaut
python3 training/train_v3_ultimate.py

# Config custom
python3 training/train_v3_ultimate.py --config config/training_config_v3.yaml
```

## 📊 Monitoring

### Logs Locaux

```bash
# Logs training
tail -f logs/training_v3.log

# Logs TensorBoard
tensorboard --logdir runs/v3_ultimate/ --port 6006

# Ouvrir dans navigateur
http://localhost:6006
```

### Weights & Biases

1. Créer compte sur [wandb.ai](https://wandb.ai)
2. Login :
   ```bash
   wandb login
   ```
3. Le training loggera automatiquement sur W&B
4. Dashboard : `https://wandb.ai/<username>/Ploutos_Trading_V3_ULTIMATE`

### Métriques Trackées

- **Equity** : Valeur totale du portfolio
- **Total Return** : Rendement cumulé
- **Total Trades** : Nombre de trades exécutés
- **Episode Reward** : Récompense par épisode
- **Policy Loss** : Perte de la policy
- **Value Loss** : Perte de la value function
- **Entropy** : Entropie (exploration)
- **KL Divergence** : Divergence KL
- **Explained Variance** : Qualité de la value function

## 📁 Structure des Fichiers

```
project_ploutos/
├── config/
│   └── training_config_v3.yaml    # Config principale
├── core/
│   ├── universal_environment_v4.py  # Env ultra-réaliste
│   └── advanced_features.py         # 50+ features
├── training/
│   └── train_v3_ultimate.py         # Script training
├── models/
│   ├── v3_checkpoints/              # Checkpoints auto
│   ├── v3_best/                     # Meilleur modèle
│   └── v3_ultimate/                 # Modèle final
├── logs/
│   ├── training_v3.log
│   └── v3_eval/                     # Logs évaluation
└── runs/
    └── v3_ultimate/                 # TensorBoard logs
```

## ⚙️ Configuration Avancée

### Modifier la Config

Éditer `config/training_config_v3.yaml` :

```yaml
training:
  total_timesteps: 20000000  # Doubler la durée
  learning_rate: 0.0001      # Learning rate plus élevé
  
environment:
  buy_pct: 0.20              # Trades plus agressifs
  max_position_pct: 0.30     # Positions plus larges
  
data:
  tickers:
    - NVDA
    - TSLA
    # Ajouter plus de tickers
  period: '10y'              # Plus de données
```

### Tickers Recommandés

**High Momentum** : NVDA, TSLA, AMD, MSTR
**Tech Stable** : MSFT, AAPL, GOOGL, META
**Indices** : SPY, QQQ, IWM, DIA
**Secteurs** : XLE, XLF, XLK, XLV, XLI

### Data Augmentation

```yaml
data:
  period: '10y'        # Maximum de données
  interval: '1h'       # Résolution horaire
  
  # Ajouter différents régimes de marché
  # Bull market : 2019-2021
  # Bear market : 2022
  # Recovery : 2023-2024
```

## 🛡️ Troubleshooting

### Out of Memory (OOM)

Réduire :
```yaml
training:
  n_envs: 8          # Au lieu de 16
  batch_size: 4096   # Au lieu de 8192
```

### Training Instable

Augmenter :
```yaml
training:
  max_grad_norm: 1.0    # Au lieu de 0.5
  target_kl: 0.02       # Au lieu de 0.015
```

### Trop d'Exploration

Réduire :
```yaml
training:
  ent_coef: 0.001   # Au lieu de 0.01
```

### Modèle Trop Conservateur

Augmenter :
```yaml
environment:
  buy_pct: 0.25              # Plus agressif
  reward_scaling: 2.0        # Amplifier rewards
```

## 🎯 Résultats Attendus

Après 10M timesteps (~3-5h sur RTX 3080) :

- **Sharpe Ratio** : > 1.5
- **Max Drawdown** : < 20%
- **Win Rate** : 55-65%
- **Avg Trade Duration** : 5-10 steps
- **Total Return** : 30-50% annualisé (backtesting)

## 🚀 Prochaines Étapes

1. **Tester le modèle** : `python3 tests/test_model_v3.py`
2. **Backtesting** : `python3 backtesting/backtest_v3.py`
3. **Déployer** : Copier le meilleur modèle vers VPS
4. **Paper Trading** : Tester en conditions réelles

## 📚 Ressources

- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Technical Analysis Library](https://technical-analysis-library-in-python.readthedocs.io/)
- [Weights & Biases](https://docs.wandb.ai/)

## ❓ Support

En cas de problème :
1. Vérifier les logs : `logs/training_v3.log`
2. Vérifier GPU : `nvidia-smi`
3. Vérifier config : `cat config/training_config_v3.yaml`

---

**Auteur** : Ploutos AI Team  
**Date** : Décembre 2025  
**Version** : V3 ULTIMATE
