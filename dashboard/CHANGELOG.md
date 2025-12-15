# Changelog - Dashboard Ploutos

## Version 2.1 - 2025-12-15

### 🆕 Nouveautés : Analyse Technique en Temps Réel

#### Indicateurs Techniques
- **RSI (Relative Strength Index)** : Détecte surachat (>70) / survente (<30)
- **MACD** : Convergence/divergence des moyennes mobiles (momentum)
- **Bollinger Bands** : Bandes de volatilité (surachat/survente)
- **Stochastic Oscillator** : Momentum comparé à la fourchette de prix
- **ATR (Average True Range)** : Mesure de volatilité (pour stop-loss)
- **OBV (On-Balance Volume)** : Flux de volume cumulatif
- **VWAP** : Prix moyen pondéré par volume
- **SMA/EMA** : Moyennes mobiles simples et exponentielles

#### Signaux de Trading Automatiques
- **Signal BUY/SELL/HOLD** avec scoring de force (0-100)
- **Stop-loss automatique** : 2x ATR sous/au-dessus du prix d'entrée
- **Take-profit automatique** : 3x ATR dans la direction du trade
- **Risk/Reward ratio** calculé automatiquement
- **Détection de tendance** : BULLISH / BEARISH / NEUTRAL
- **Confiance du signal** : Score de 0.0 à 1.0
- **Raisons détaillées** : Liste explicative de chaque décision

### 🔧 Architecture Zéro Régression

#### Import Lazy (Sécurité Maximale)
```python
# Le module technique est importé uniquement si appelé
# Dashboard démarre TOUJOURS même si yfinance absent
# Erreur propre 503 uniquement sur endpoints /api/technical/*
```

#### Compatibilité Totale
- ✅ **TOUS les endpoints existants fonctionnent**
- ✅ **PostgreSQL avec fallback JSON** (inchangé)
- ✅ **Client Alpaca** (inchangé)
- ✅ **Analytics avancés** (inchangés)
- ✅ **Pages HTML** (inchangées)
- ✅ **WebSocket** (inchangé)

### 🚀 Nouveaux Endpoints API

#### 1. Analyse Complète
```bash
GET /api/technical/<SYMBOL>?period=3mo&interval=1h
```

**Exemple :**
```bash
curl "http://localhost:5000/api/technical/NVDA?period=1y&interval=1d"
```

**Réponse :**
```json
{
  "success": true,
  "symbol": "NVDA",
  "timestamp": "2025-12-15T14:30:00",
  "period": "1y",
  "interval": "1d",
  "indicators": {
    "price": {
      "current": 485.23,
      "change_24h": 2.45,
      "high_24h": 492.10,
      "low_24h": 478.50
    },
    "moving_averages": {
      "sma_20": 480.12,
      "sma_50": 465.34,
      "ema_20": 482.56
    },
    "macd": {
      "macd_line": 3.45,
      "signal_line": 2.10,
      "histogram": 1.35
    },
    "momentum": {
      "rsi": 68.5,
      "stochastic_k": 75.2,
      "stochastic_d": 72.8
    },
    "volatility": {
      "bb_upper": 495.30,
      "bb_middle": 480.12,
      "bb_lower": 464.94,
      "atr": 8.45
    },
    "volume": {
      "obv": 145230000,
      "vwap": 482.90,
      "volume_24h": 32500000
    }
  },
  "trading_signal": {
    "signal": "BUY",
    "strength": 71,
    "trend": "BULLISH",
    "confidence": 0.71,
    "reasons": [
      "Prix au-dessus SMA 20 et 50 (tendance haussière)",
      "MACD croisement haussier",
      "Volume confirmant la hausse"
    ],
    "entry_price": 485.23,
    "stop_loss": 468.33,
    "take_profit": 510.58,
    "risk_reward_ratio": 1.5
  }
}
```

#### 2. Signal Rapide
```bash
GET /api/technical/<SYMBOL>/signal?period=3mo&interval=1h
```

**Exemple :**
```bash
curl "http://localhost:5000/api/technical/AAPL/signal"
```

**Réponse :**
```json
{
  "success": true,
  "symbol": "AAPL",
  "timestamp": "2025-12-15T14:30:00",
  "signal": "HOLD",
  "strength": 50,
  "trend": "NEUTRAL",
  "confidence": 0.5,
  "entry_price": 195.67,
  "stop_loss": 192.34,
  "take_profit": 200.99,
  "reasons": [
    "RSI neutre (52.3)",
    "Signaux mixtes, attendre confirmation"
  ]
}
```

#### 3. Analyse Batch (Plusieurs Symboles)
```bash
POST /api/technical/batch
Content-Type: application/json

{
  "symbols": ["NVDA", "MSFT", "AAPL"],
  "period": "3mo",
  "interval": "1h"
}
```

**Exemple :**
```bash
curl -X POST http://localhost:5000/api/technical/batch \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["NVDA", "MSFT", "AAPL"], "period": "6mo"}'
```

**Réponse :**
```json
{
  "success": true,
  "timestamp": "2025-12-15T14:30:00",
  "results": {
    "NVDA": {
      "signal": "BUY",
      "strength": 71,
      "trend": "BULLISH",
      "confidence": 0.71,
      "entry_price": 485.23,
      "reasons": [
        "Prix au-dessus SMA 20 et 50",
        "MACD croisement haussier",
        "Volume confirmant la hausse"
      ]
    },
    "MSFT": {
      "signal": "HOLD",
      "strength": 50,
      "trend": "NEUTRAL",
      "confidence": 0.5,
      "entry_price": 372.45,
      "reasons": ["Signaux mixtes"]
    },
    "AAPL": {
      "signal": "SELL",
      "strength": 57,
      "trend": "BEARISH",
      "confidence": 0.57,
      "entry_price": 195.67,
      "reasons": [
        "RSI suracheté (72.1)",
        "Prix au-dessus bande de Bollinger supérieure"
      ]
    }
  }
}
```

#### 4. Scan Watchlist Complète
```bash
GET /api/technical/watchlist?period=3mo&interval=1h
```

**Exemple :**
```bash
curl "http://localhost:5000/api/technical/watchlist?period=1mo&interval=1d"
```

**Réponse :**
```json
{
  "success": true,
  "timestamp": "2025-12-15T14:30:00",
  "total_symbols": 18,
  "buy_signals_count": 6,
  "sell_signals_count": 3,
  "top_buy_opportunities": {
    "NVDA": {"signal": "BUY", "strength": 71, "confidence": 0.71},
    "MSFT": {"signal": "BUY", "strength": 64, "confidence": 0.64},
    "AMZN": {"signal": "BUY", "strength": 57, "confidence": 0.57}
  },
  "top_sell_signals": {
    "XOM": {"signal": "SELL", "strength": 68, "confidence": 0.68},
    "CVX": {"signal": "SELL", "strength": 60, "confidence": 0.60}
  },
  "all_results": {
    "NVDA": {...},
    "MSFT": {...},
    ...
  }
}
```

### 📚 Paramètres Disponibles

#### Paramètres `period` (période historique)
- `1mo` : 1 mois
- `3mo` : 3 mois (🟢 **défaut**)
- `6mo` : 6 mois
- `1y` : 1 an
- `2y` : 2 ans

#### Paramètres `interval` (granularité)
- `1m` : 1 minute (uniquement pour period < 7 jours)
- `5m` : 5 minutes
- `15m` : 15 minutes
- `1h` : 1 heure (🟢 **défaut**)
- `1d` : 1 jour

### 💬 Interprétation des Signaux

#### Force du Signal (strength)
- **0-30** : Signal très faible, ne pas trader
- **30-50** : Signal faible, attendre confirmation
- **50-70** : Signal modéré, considérer le trade
- **70-85** : 🟢 Signal fort, bon moment pour entrer
- **85-100** : Signal très fort, opportunité majeure

#### Tendance (trend)
- **BULLISH** : 🟢 Tendance haussière confirmée (bon pour BUY)
- **BEARISH** : 🔴 Tendance baissière confirmée (bon pour SELL/short)
- **NEUTRAL** : 🟡 Pas de tendance claire (rester à l'écart)

#### Confiance (confidence)
- **< 0.3** : Très faible, ignorer
- **0.3 - 0.5** : Faible, surveiller
- **0.5 - 0.7** : Modérée, considérer
- **0.7 - 0.85** : 🟢 Forte, trader
- **> 0.85** : Très forte, opportunité exceptionnelle

### 📦 Installation (Strict Nécessaire)

#### Dépendances Supplémentaires
```bash
pip install yfinance>=0.2.0
# pandas et numpy déjà installés pour analytics.py
```

#### Vérification Installation
```bash
# Tester que le dashboard démarre (même si yfinance absent)
cd /root/ploutos/project_ploutos
python dashboard/app_v2.py

# Vérifier health check
curl http://localhost:5000/api/health
# Réponse : {"features": {"technical_analysis": true/false}}

# Tester un signal (si yfinance présent)
curl http://localhost:5000/api/technical/NVDA/signal
```

### ⚠️  Zéro Régression Garantie

#### Si `yfinance` N'EST PAS Installé
- ✅ Dashboard **démarre normalement**
- ✅ Tous les endpoints existants **fonctionnent** (/api/account, /api/positions, etc.)
- ✅ Analytics avancés **fonctionnent** (/api/analytics/advanced)
- ❌ Endpoints `/api/technical/*` retournent **503 Service Unavailable** (propre)

#### Si `yfinance` EST Installé
- ✅ **Toutes les fonctionnalités** disponibles
- ✅ Endpoints technique **fonctionnels**
- ✅ Aucun impact sur performance des autres endpoints

### 🐛 Gestion des Erreurs

#### Erreur 503 : Analyse Technique Indisponible
```json
{
  "success": false,
  "error": "Analyse technique indisponible sur ce serveur",
  "details": "Dépendances manquantes: No module named 'yfinance'"
}
```

**Solution :** `pip install yfinance`

#### Erreur 500 : Symbole Invalide ou Yahoo Inaccessible
```json
{
  "success": false,
  "error": "Pas de données pour INVALID_SYMBOL"
}
```

**Causes possibles :**
- Ticker invalide
- Yahoo Finance temporairement inaccessible
- Pas de données historiques pour ce symbole

### 📝 Fichiers Modifiés (Version 2.1)

```
dashboard/
├── technical_analysis.py    # NOUVEAU : Module analyse technique
├── app_v2.py                 # MODIFIÉ : Ajout 4 endpoints + import lazy
└── CHANGELOG.md             # MODIFIÉ : Documentation v2.1
```

**Aucun fichier supprimé ou renommé**

### 🧪 Tests de Non-Régression

#### Test 1 : Dashboard Démarre
```bash
python dashboard/app_v2.py
# Doit afficher : "✅ Dashboard v2.1 prêt sur http://0.0.0.0:5000"
```

#### Test 2 : Endpoints Existants OK
```bash
curl http://localhost:5000/api/account
curl http://localhost:5000/api/positions
curl http://localhost:5000/api/analytics/advanced
# Tous doivent retourner success:true
```

#### Test 3 : Nouveaux Endpoints (Si yfinance présent)
```bash
curl http://localhost:5000/api/technical/NVDA/signal
# Doit retourner un signal BUY/SELL/HOLD
```

---

## Version 2.0 - 2025-12-09

### 🎉 Nouveautés

#### Métriques Financières Avancées
- **Sharpe Ratio** : Mesure du rendement ajusté au risque (annualisé)
- **Sortino Ratio** : Comme Sharpe mais pénalise uniquement la volatilité baissière
- **Calmar Ratio** : Rapport rendement annualisé / max drawdown
- **Max Drawdown** : Baisse maximale du portfolio avec dates début/fin
- **Profit Factor** : Ratio gains moyens / pertes moyennes
- **Win Rate avancé** : Analyse des paires BUY->SELL rentables

#### Analytics par Symbole
- Statistiques détaillées par ticker
- Historique des trades filtrés
- Volume et prix moyens

#### Architecture Améliorée
- **Connexion PostgreSQL native** avec fallback automatique sur JSON
- **Module analytics.py** dédié aux calculs financiers
- **Pandas/Numpy** pour calculs performants
- **Gestion d'erreurs robuste** avec logging détaillé

### 🔧 Technique

#### Nouveaux Fichiers
```
dashboard/
├── app.py              # Version 2.0 (remplace l'ancien)
├── app_legacy.py       # Ancien dashboard (backup automatique)
├── analytics.py        # Module de calculs financiers
├── requirements_v2.txt # Dépendances v2
└── CHANGELOG.md        # Ce fichier

scripts/
└── patch_dashboard_v2.sh  # Script de déploiement
```

#### Nouveaux Endpoints

**Analytics Avancés**
```bash
# Métriques avancées (Sharpe, Sortino, etc.)
GET /api/analytics/advanced?days=30

# Analytics pour un symbole
GET /api/analytics/symbol/<SYMBOL>?days=30

# Health check système
GET /api/health
```

### 📦 Dépendances Ajoutées

```txt
numpy>=1.24.0
pandas>=2.0.0
psycopg2-binary==2.9.9
```

### 🚀 Installation

#### Méthode Automatique (Recommandée)
```bash
cd /root/ploutos/project_ploutos
bash scripts/patch_dashboard_v2.sh
```

Le script :
- ✅ Crée un backup automatique
- ✅ Installe le nouveau dashboard
- ✅ Vérifie PostgreSQL
- ✅ Teste les imports Python
- ✅ Redémarre le service si nécessaire
- ✅ Rollback automatique en cas d'erreur

---

## Version 1.0 - 2025-11-XX

### Fonctionnalités Initiales
- Dashboard Flask basique
- Lecture trades depuis JSON
- Affichage positions Alpaca
- Stats simples (buy/sell count)
- WebSocket pour temps réel

---

**Généré le** : 2025-12-15  
**Auteur** : Vimif  
**Projet** : Ploutos Trading IA  
**Branche** : feature/v7-predictive-models  
