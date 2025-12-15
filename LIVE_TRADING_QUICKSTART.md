# 🚀 LIVE TRADING DASHBOARD - GUIDE DE DÉMARRAGE RAPIDE

## 🎯 Vue d'ensemble

Système complet de détection de signaux BUY/SELL en temps réel via Alpaca WebSocket avec dashboard web intégré.

---

## 📚 Composants créés

### **Backend**
1. `streaming/signal_detector.py` - Détecteur de signaux (5 stratégies)
2. `streaming/live_analyzer.py` - Analyseur temps réel WebSocket
3. `web/routes/live_trading.py` - API Flask + SSE

### **Frontend**
1. `web/templates/live.html` - Dashboard interactif
2. `web/templates/components/nav.html` - Navigation

### **Documentation**
1. `streaming/README.md` - Guide complet
2. Ce fichier - Quick start

---

## ⚡ Installation Express

### **1. Installer les dépendances**

```bash
cd /root/ploutos/project_ploutos

# Activer virtualenv si nécessaire
source venv/bin/activate  # ou ton virtualenv

# Installer alpaca-py
pip install alpaca-py pandas numpy

# Installer TA-Lib (pour indicateurs techniques)
sudo apt-get update
sudo apt-get install -y libta-lib0-dev
pip install TA-Lib

# Alternative si problèmes avec TA-Lib
pip install pandas_ta  # À la place de TA-Lib
```

---

### **2. Configuration Alpaca**

Édite `config/settings.py` (ou crée-le) :

```python
import os

# Alpaca API (Paper Trading)
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', 'ton_api_key_ici')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', 'ton_secret_key_ici')
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
```

**Où trouver tes clés :**
1. Va sur [alpaca.markets](https://alpaca.markets)
2. Crée un compte Paper Trading (gratuit)
3. Dashboard → API Keys → Generate New Key
4. Copie API Key + Secret Key

**Variables d'environnement (recommandé) :**

```bash
# Ajoute dans ~/.bashrc ou ~/.zshrc
export ALPACA_API_KEY="PKxxxxxxxxxxxxx"
export ALPACA_SECRET_KEY="xxxxxxxxxxxxxxxxxxxxxxxx"

# Recharge
source ~/.bashrc
```

---

### **3. Test du système**

#### **Test 1 : SignalDetector (standalone)**

```bash
python streaming/signal_detector.py
```

**Sortie attendue :**
```
✅ Test SignalDetector OK
Signal: BUY, Confidence: 75%
```

#### **Test 2 : LiveAnalyzer (console)**

```bash
python streaming/live_analyzer.py
```

**Sortie attendue :**
```
🚀 Démarrage du Live Analyzer...
Tickers surveillés: NVDA, AAPL, MSFT, GOOGL, TSLA
Timeframe: 1 minute(s)
[14:23:45] NVDA: $520.45 | Vol: 2,345,678
```

**Arrêter** : `Ctrl+C`

---

### **4. Démarrer le Dashboard**

```bash
cd /root/ploutos/project_ploutos
python web/app.py
```

**Sortie attendue :**
```
================================================================================
🌐 PLOUTOS - V8 ORACLE + TRADER PRO + 5 TOOLS + WATCHLISTS + LIVE TRADING
================================================================================

🚀 http://0.0.0.0:5000
🔥 LIVE TRADING: /live
...
```

---

## 🎮 Utilisation du Dashboard

### **1. Accéder au dashboard**

Ouvre ton navigateur :
```
http://localhost:5000/live
```

---

### **2. Démarrer le monitoring**

1. **Tickers** : Entre les tickers à surveiller (ex: `NVDA,AAPL,TSLA`)
2. **Timeframe** : Choisis 1, 5 ou 15 minutes
3. Clique sur **"Démarrer"**

👉 Le système commence à recevoir les barres en temps réel d'Alpaca

---

### **3. Observer les signaux**

Les signaux BUY/SELL apparaissent automatiquement dans le feed quand :
- **≥ 3 stratégies** alignent dans la même direction
- **Confiance ≥ 60%**

**Exemple de signal :**
```
🚨 SIGNAL BUY DÉTECTÉ !
Ticker: NVDA
Prix: $521.20
Confiance: 75%

Raisons:
  - RSI survente (28.5)
  - EMA Golden Cross
  - MACD bullish
  - Volume spike (1.8x)
```

---

### **4. Arrêter le monitoring**

Clique sur **"Arrêter"** pour stopper le flux.

Les statistiques de session s'affichent dans les logs du backend.

---

## 📊 API Endpoints

Le dashboard utilise ces endpoints (disponibles aussi en direct) :

### **Démarrer**
```bash
curl -X POST http://localhost:5000/api/live/start \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["NVDA", "AAPL", "TSLA"],
    "timeframe": 1
  }'
```

### **État actuel**
```bash
curl http://localhost:5000/api/live/state
```

### **Arrêter**
```bash
curl -X POST http://localhost:5000/api/live/stop
```

### **Stream SSE (temps réel)**
```bash
curl -N http://localhost:5000/api/live/stream
```

---

## ⚙️ Configuration avancée

### **Modifier les stratégies de détection**

Édite `streaming/signal_detector.py` :

```python
# Ligne ~150
if rsi < 30:  # Modifier le seuil RSI
    reasons.append(f"RSI survente ({rsi:.1f})")
    points += 20  # Modifier le poids
```

### **Ajouter des indicateurs**

```python
# Dans signal_detector.py, méthode _calculate_indicators

# Exemple : Ajouter VWAP
vwap = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
self.indicators['vwap'] = vwap.iloc[-1]
```

### **Changer le seuil de confiance**

Édite `streaming/live_analyzer.py` :

```python
# Ligne ~85
if signal["signal"] in ["BUY", "SELL"] and signal["confidence"] >= 60:
    # Changer 60 par 70 pour être plus sélectif
```

---

## 🐞 Débogage

### **Problème : "LiveAnalyzer non disponible"**

**Solution :**
```bash
# Vérifier les imports
python -c "from streaming.live_analyzer import LiveAnalyzer; print('OK')"

# Si erreur, installer alpaca-py
pip install alpaca-py
```

---

### **Problème : "Pas de données reçues"**

**Causes possibles :**
1. **Marché fermé** → Alpaca IEX ne transmet que pendant heures de marché (9h30-16h00 ET)
2. **Mauvaises clés API** → Vérifie `ALPACA_API_KEY` et `ALPACA_SECRET_KEY`
3. **Ticker invalide** → Utilise uniquement des actions US (pas de crypto/ETF exotiques)

**Vérifier connexion Alpaca :**
```python
from alpaca.data.live import StockDataStream
from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY

stream = StockDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY, feed='iex')
print("✅ Connexion OK")
```

---

### **Problème : "ImportError: No module named 'talib'"**

**Solution 1 : Installer TA-Lib (recommandé)**
```bash
sudo apt-get install -y libta-lib0-dev
pip install TA-Lib
```

**Solution 2 : Utiliser pandas_ta**
```bash
pip install pandas_ta
```

Puis édite `streaming/signal_detector.py` :
```python
# Remplacer
import talib
# Par
import pandas_ta as ta
```

---

## 📈 Prochaines étapes

### **1. Notifications Telegram**

Ajoute ce callback dans `live_analyzer.py` :

```python
import requests

def send_telegram(signal):
    bot_token = "ton_bot_token"
    chat_id = "ton_chat_id"
    
    message = f"""
🚨 SIGNAL {signal['signal']} DÉTECTÉ !

Ticker: {signal['ticker']}
Prix: ${signal['current_price']:.2f}
Confiance: {signal['confidence']}%
"""
    
    requests.post(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        json={"chat_id": chat_id, "text": message}
    )

analyzer.add_signal_callback(send_telegram)
```

---

### **2. Logging en base de données**

Crée la table PostgreSQL :

```sql
CREATE TABLE live_signals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    signal VARCHAR(10),
    confidence INTEGER,
    price DECIMAL(10, 2),
    reasons JSONB,
    indicators JSONB,
    timestamp TIMESTAMP DEFAULT NOW()
);
```

Puis utilise le callback dans `streaming/README.md` (section "Logging en BDD").

---

### **3. Exécution automatique d'ordres**

⚠️ **ATTENTION : Utilise Paper Trading d'abord !**

```python
from trading.alpaca_client import AlpacaClient

def auto_trade(signal):
    if signal['confidence'] < 80:
        return  # Ignorer signaux faibles
    
    client = AlpacaClient(paper_trading=True)  # TOUJOURS True au début !
    
    if signal['signal'] == "BUY":
        client.place_market_order(
            ticker=signal['ticker'],
            notional=500,  # $500
            side='buy'
        )
        print(f"✅ Ordre BUY {signal['ticker']} exécuté")

analyzer.add_signal_callback(auto_trade)
```

---

## 📚 Ressources

- **Documentation complète** : `streaming/README.md`
- **Alpaca Docs** : [docs.alpaca.markets](https://docs.alpaca.markets)
- **TA-Lib** : [ta-lib.org](https://ta-lib.org)
- **Stable-Baselines3** : [stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io)

---

## ❓ Support

Problème ? Ouvre une issue GitHub ou contacte-moi directement.

**Auteur** : Thomas BOISAUBERT  
**Projet** : Ploutos V8 Oracle  
**Date** : Décembre 2025  

---

🚀 **Happy Trading !**
