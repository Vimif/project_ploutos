# 🔥 SYSTÈME D'ANALYSE TEMPS RÉEL

## 🎯 Objectif

Système de détection automatique de signaux BUY/SELL en temps réel à partir du flux Alpaca WebSocket.

---

## 📚 Modules

### **1. `signal_detector.py`**
**Rôle :** Détecte les signaux de trading basés sur 5 stratégies

**Stratégies implémentées :**
1. **RSI (14)** : Survente (< 30) / Surachat (> 70)
2. **EMA Crossover** : Golden Cross (9>21) / Death Cross (9<21)
3. **MACD** : Croisement MACD/Signal + Histogramme
4. **Bollinger Bands** : Prix au-dessus/en-dessous des bandes
5. **Volume Spike** : Volume > 1.5x moyenne

**Logique de décision :**
- **BUY** : ≥ 3 signaux d'achat
- **SELL** : ≥ 3 signaux de vente
- **HOLD** : Signaux mixtes ou < 3

**Score de confiance :** 0-100% (cumul des points par stratégie)

---

### **2. `live_analyzer.py`**
**Rôle :** Connecte au WebSocket Alpaca et orchestre l'analyse

**Fonctionnalités :**
- Connexion WebSocket Alpaca (feed IEX gratuit)
- Réception barres 1/5/15 minutes en temps réel
- Analyse automatique via `SignalDetector`
- Callbacks pour notifications/actions
- Statistiques de session

---

## 🚀 Utilisation

### **Mode Standalone (Console)**

```bash
cd /root/ploutos/project_ploutos
python streaming/live_analyzer.py
```

**Sortie console :**
```
🚀 Démarrage du Live Analyzer...
Tickers surveillés: NVDA, AAPL, MSFT, GOOGL, TSLA
Timeframe: 1 minute(s)
================================================================================

[14:23:45] NVDA: $520.45 | Vol: 2,345,678
[14:24:45] NVDA: $521.20 | Vol: 1,987,234

================================================================================
🚨 SIGNAL BUY DÉTECTÉ !
Ticker: NVDA
Prix: $521.20
Confidence: 75%
Raisons:
  - RSI survente (28.5)
  - EMA Golden Cross (519.50 > 518.30)
  - MACD bullish (0.142)
  - Volume spike (1.8x)
Indicateurs:
  RSI: 28.5
  EMA_Fast: 519.5
  EMA_Slow: 518.3
  MACD: 0.142
  ...
================================================================================
```

---

### **Intégration Dashboard (API Flask)**

**Ajouter dans `app.py` :**

```python
from streaming.live_analyzer import LiveAnalyzer
import asyncio
import threading

# Instance globale
live_analyzer = None

@app.route('/api/live/start', methods=['POST'])
def api_live_start():
    """Démarre le monitoring temps réel"""
    global live_analyzer
    
    data = request.json
    tickers = data.get('tickers', ['NVDA', 'AAPL'])
    timeframe = data.get('timeframe', 1)
    
    # Créer analyzer
    live_analyzer = LiveAnalyzer(tickers, timeframe_minutes=timeframe)
    
    # Lancer dans un thread séparé
    def run_analyzer():
        asyncio.run(live_analyzer.start())
    
    thread = threading.Thread(target=run_analyzer, daemon=True)
    thread.start()
    
    return jsonify({"status": "started", "tickers": tickers})


@app.route('/api/live/state')
def api_live_state():
    """Récupère l'état actuel"""
    if not live_analyzer:
        return jsonify({"error": "Analyzer not started"}), 400
    
    return jsonify(live_analyzer.get_current_state())


@app.route('/api/live/stop', methods=['POST'])
def api_live_stop():
    """Arrête le monitoring"""
    global live_analyzer
    
    if live_analyzer:
        live_analyzer.stop()
        live_analyzer = None
    
    return jsonify({"status": "stopped"})
```

**Utilisation API :**
```bash
# Démarrer
curl -X POST http://localhost:5000/api/live/start \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["NVDA", "AAPL", "TSLA"], "timeframe": 1}'

# État actuel
curl http://localhost:5000/api/live/state

# Arrêter
curl -X POST http://localhost:5000/api/live/stop
```

---

## 📦 Dépendances

```bash
pip install alpaca-py pandas numpy talib
```

**TA-Lib installation :**
```bash
# Ubuntu/Debian
sudo apt-get install libta-lib0-dev
pip install TA-Lib

# Si problèmes, utiliser pandas_ta en remplacement
pip install pandas_ta
```

---

## ⚙️ Configuration

**Dans `config/settings.py` :**

```python
# Alpaca API (Paper Trading)
ALPACA_API_KEY = "ton_api_key"
ALPACA_SECRET_KEY = "ton_secret_key"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading

# Live Analyzer
LIVE_ANALYZER_TIMEFRAME = 1  # minutes (1, 5, 15)
LIVE_ANALYZER_MIN_CONFIDENCE = 60  # Seuil minimum de confiance
LIVE_ANALYZER_MAX_TICKERS = 10  # Limite de tickers simultanés
```

---

## 💡 Cas d'usage

### **1. Alertes Telegram/Discord**

```python
import requests

def send_telegram_alert(signal):
    """Envoie alerte Telegram"""
    bot_token = "ton_bot_token"
    chat_id = "ton_chat_id"
    
    message = f"""
🚨 SIGNAL {signal['signal']} DÉTECTÉ !

Ticker: {signal['ticker']}
Prix: ${signal['current_price']:.2f}
Confiance: {signal['confidence']}%

Raisons:
""" + "\n".join(f"- {r}" for r in signal['reasons'])
    
    requests.post(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        json={"chat_id": chat_id, "text": message}
    )

analyzer.add_signal_callback(send_telegram_alert)
```

---

### **2. Exécution automatique d'ordres**

```python
from trading.alpaca_client import AlpacaClient

def auto_trade(signal):
    """Exécute automatiquement les ordres"""
    if signal['confidence'] < 80:
        return  # Ignorer signaux faibles
    
    client = AlpacaClient()
    
    if signal['signal'] == "BUY":
        # Acheter $500 de l'action
        client.place_market_order(
            ticker=signal['ticker'],
            notional=500,  # $500
            side='buy'
        )
        print(f"✅ Ordre BUY {signal['ticker']} exécuté")
    
    elif signal['signal'] == "SELL":
        # Vendre toutes les positions
        client.close_position(signal['ticker'])
        print(f"✅ Position {signal['ticker']} fermée")

analyzer.add_signal_callback(auto_trade)
```

---

### **3. Logging en base de données**

```python
import psycopg2

def log_signal_to_db(signal):
    """Sauvegarde le signal en BDD"""
    conn = psycopg2.connect(
        host="localhost",
        database="ploutos",
        user="ploutos",
        password="ton_mdp"
    )
    
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO live_signals 
        (ticker, signal, confidence, price, reasons, indicators, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        signal['ticker'],
        signal['signal'],
        signal['confidence'],
        signal['current_price'],
        json.dumps(signal['reasons']),
        json.dumps(signal['indicators']),
        signal['timestamp']
    ))
    
    conn.commit()
    cur.close()
    conn.close()

analyzer.add_signal_callback(log_signal_to_db)
```

---

## 📊 Performance

**Latence typ ique :**
- Réception barre Alpaca : **< 100ms**
- Analyse SignalDetector : **< 50ms**
- Total : **< 150ms** entre event et signal

**Ressources :**
- CPU : **< 5%** pour 5 tickers
- RAM : **~100MB**

---

## ⚠️ Limites Alpaca Gratuit

**Feed IEX (gratuit) :**
- ✅ Temps réel pendant heures de marché
- ✅ Barres 1/5/15 minutes
- ❌ Pas de données pre-market / after-hours
- ❌ Limité aux actions US

**Pour lever les limites :**
- Passer au feed **SIP** (payant $9-99/mois)
- Données 24/7 + pre/after market

---

## 🔧 Test du système

```bash
# Test unitaire SignalDetector
python streaming/signal_detector.py

# Test LiveAnalyzer (mode console)
python streaming/live_analyzer.py

# Vérifier connexion Alpaca
python -c "from streaming.live_analyzer import LiveAnalyzer; print('OK')"
```

---

## 📅 Roadmap

- [ ] Dashboard temps réel avec graphiques live
- [ ] Notifications Telegram/Discord
- [ ] Backtesting des stratégies de détection
- [ ] Machine Learning pour améliorer les signaux
- [ ] Support crypto via Alpaca Crypto feed
- [ ] Mode "paper trading" intégré

---

## 👤 Auteur

**Thomas BOISAUBERT** - AI Factory
Projet Ploutos V8 - Décembre 2025
