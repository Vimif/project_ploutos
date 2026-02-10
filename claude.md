# Ploutos Trading Bot — Guide IA

> Ce fichier sert de référence rapide pour tout assistant IA travaillant sur ce projet.

## Vue d'ensemble

Ploutos est un bot de trading algorithmique basé sur le **Reinforcement Learning** (PPO via stable-baselines3). Il observe les marchés US (NYSE/NASDAQ), calcule des features techniques avancées, et prend des décisions BUY/HOLD/SELL de manière autonome.

**Statut actuel** : Paper trading uniquement. Pas encore prêt pour le trading réel.

---

## Architecture

```
project_ploutos/
├── config/                 # Configs YAML + dataclass Python
│   ├── autonomous_config.yaml
│   ├── training_config_v6_better_timing.yaml
│   ├── training_config_v6_extended_50m.yaml
│   └── config.py           # Dataclasses PloutosConfig
├── core/                   # Logique métier
│   ├── universal_environment_v6_better_timing.py  # Env Gym actif (V6)
│   ├── advanced_features_v2.py    # 60+ features/ticker
│   ├── data_fetcher.py            # Multi-source: Alpaca → yfinance → Polygon
│   ├── risk_manager.py            # Kelly, Sharpe, drawdown, position sizing
│   ├── market_analyzer.py         # RSI, MACD, Bollinger, analyse trend
│   ├── market_status.py           # Vérifie si NYSE/NASDAQ ouvert
│   ├── self_improvement.py        # Analyse trades perdants, suggestions
│   ├── trading_callback.py        # Callback W&B pour training metrics
│   └── transaction_costs.py       # Slippage, market impact, latence
├── trading/                # Connexion brokers
│   ├── broker_interface.py        # ABC commune
│   ├── broker_factory.py          # Factory: 'etoro' | 'alpaca'
│   ├── etoro_client.py            # Client eToro API
│   └── alpaca_client.py           # Client Alpaca API
├── training/               # Scripts d'entraînement
│   ├── train_v6_better_timing.py  # Entraînement V6 (15M steps)
│   ├── train_v6_extended_50m.py   # Entraînement V6 étendu (50M steps)
│   ├── trainer.py                 # Trainer générique
│   └── curriculum_trainer.py      # Curriculum learning
├── scripts/                # Points d'entrée CLI
│   ├── run_trader_v6.py           # ★ Script trading live/paper
│   ├── backtest_v6.py             # Backtest V6
│   ├── backtest_reliability.py    # Backtest multi-env
│   └── analyze_why_fails_v6.py    # Diagnostic timing
├── data/models/            # Modèles entraînés
│   ├── brain_tech.zip      # 68 MB — modèle principal (Tech stocks)
│   ├── brain_crypto.zip    # 7.6 MB
│   ├── brain_defensive.zip # 7.6 MB
│   └── brain_energy.zip    # 7.6 MB
├── dashboard/              # Dashboard Flask (analytics, technical)
└── docs/                   # 15 fichiers de documentation
```

---

## Conventions & patterns importants

### Versioning des environnements
- Le projet a traversé V2 → V3 → V4 → V6 (pas de V5).
- **L'environnement actif est V6** : `core/universal_environment_v6_better_timing.py`.
- Les scripts doivent utiliser `UniversalTradingEnvV6BetterTiming`.

### Tickers standards (15)
```python
TICKERS = ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'TSLA',
           'SPY', 'QQQ', 'VOO', 'VTI', 'XLE', 'XLF', 'XLK', 'XLV']
```

### Credentials (fichier `.env`)
- **eToro** : `ETORO_SUBSCRIPTION_KEY`, `ETORO_USERNAME`, `ETORO_PASSWORD`
- **Alpaca** : `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` (paper: `ALPACA_PAPER_*`)
- Le broker par défaut est `etoro` (configurable via env var `BROKER`)

### Modèles
- Format : ZIP contenant modèle PPO (stable-baselines3)
- Stockés dans `data/models/`
- Le modèle principal est `brain_tech.zip`

### Windows
- Logs et stdout doivent utiliser UTF-8 (`sys.stdout.reconfigure(encoding='utf-8')`)
- Les emojis dans les logs nécessitent ce fix

---

## Problèmes connus

### 1. BUY Timing (CRITIQUE)
Le modèle V4 achète trop tard : **85% buy-high**. V6 BetterTiming vise à corriger ce problème via les Features V2 (support/resistance, mean reversion, divergences RSI).

### 2. Manque de backtests rigoureux
Pas de walk-forward validation systématique. Les résultats de backtest (Sharpe ~1.5, win rate 55%) sont mesurés sur des périodes courtes.

### 3. Pas de stop-loss / take-profit automatiques
Le risk manager calcule des métriques mais les SL/TP ne sont pas intégrés dans l'exécution live.

### 4. Self-improvement non connecté
`core/self_improvement.py` existe mais n'est pas appelé dans la boucle de trading.

### 5. Dashboard V2 non finalisé
`dashboard/app_v2.py` (27KB) existe mais n'est pas relié au trading live.

---

## Commandes utiles

```bash
# Entraîner (V6 BetterTiming, 15M steps)
python training/train_v6_better_timing.py

# Backtest d'un modèle
python scripts/backtest_v6.py --model data/models/brain_tech.zip --days 90

# Trader en paper mode
python scripts/run_trader_v6.py --paper --broker etoro --model data/models/brain_tech.zip

# Analyser pourquoi le modèle échoue
python scripts/analyze_why_fails_v6.py --model data/models/brain_tech.zip
```

---

## Améliorations prioritaires

1. **Valider les résultats V6** — Backtester `brain_tech.zip` et mesurer le % de buy-low vs buy-high
2. **Stop-loss / Take-profit** — Intégrer SL/TP dans la boucle de trading live
3. **Walk-forward validation** — Tester sur fenêtres glissantes (pas uniquement les N derniers jours)
4. **Reconnecter le self-improvement** — Appeler `SelfImprovementEngine` dans la boucle
5. **Boucle de re-training automatique** — Si performance dégradée, relancer l'entraînement
6. **Multi-timeframe** — Combiner signaux 1h + 4h + daily
7. **Transformer architecture** — Remplacer MLP par un modèle séquentiel
