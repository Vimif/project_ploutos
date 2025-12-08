# 🌐 PLOUTOS WEB DASHBOARD

Dashboard Web moderne pour monitorer votre bot de trading en temps réel.

## ✨ Features

- 📊 **Temps réel** : Portfolio, positions, trades
- 🧠 **Auto-amélioration** : Health Score, problèmes, suggestions
- 📈 **Graphiques** : Évolution portfolio, performances
- 🎯 **Métriques** : Win rate, Sharpe, Drawdown, Profit Factor
- 🔔 **Alertes** : Notifications visuelles
- 🛡️ **Sécurité** : Accès VPN uniquement recommandé

---

## 🚀 Quick Start

### 1. Installation

```bash
cd /root/ploutos/project_ploutos

# Dépendances déjà installées via requirements.txt
pip install flask flask-cors
```

### 2. Lancement

```bash
# Lancer le dashboard
cd web
python3 app.py

# Dashboard disponible sur:
# http://localhost:5000
# ou http://VPS_IP:5000 (si configuré)
```

### 3. Accès

**Local (sur VPS)** :
```bash
curl http://localhost:5000/api/health
```

**Distant (depuis PC)** :
- Via VPN : `http://VPS_IP:5000`
- Via tunnel SSH : `ssh -L 5000:localhost:5000 root@VPS_IP`

---

## ⚙️ Configuration

### Variables d'environnement

Créer `.env` dans `web/` :

```bash
# Dashboard config
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=5000
DASHBOARD_DEBUG=false

# Alpaca (hérité de la racine)
ALPACA_PAPER_API_KEY=your_key
ALPACA_PAPER_SECRET_KEY=your_secret
```

### Lancement comme service

Créer `/etc/systemd/system/ploutos-dashboard.service` :

```ini
[Unit]
Description=Ploutos Web Dashboard
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/ploutos/project_ploutos/web
Environment="PATH=/root/ai-factory/venv/bin"
ExecStart=/root/ai-factory/venv/bin/python3 app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Activer :
```bash
sudo systemctl daemon-reload
sudo systemctl enable ploutos-dashboard
sudo systemctl start ploutos-dashboard
sudo systemctl status ploutos-dashboard
```

---

## 🔌 API Endpoints

### Status
```bash
GET /api/status
# Réponse: {status, timestamp, alpaca_connected, self_improvement_available}
```

### Account
```bash
GET /api/account
# Réponse: {cash, portfolio_value, buying_power, equity, ...}
```

### Positions
```bash
GET /api/positions
# Réponse: [{symbol, qty, market_value, unrealized_pl, ...}, ...]
```

### Trades
```bash
GET /api/trades?days=7
# Réponse: [{timestamp, symbol, action, price, amount, ...}, ...]
```

### Performance
```bash
GET /api/performance?days=7
# Réponse: {total_trades, win_count, total_invested, net_pnl, ...}
```

### Auto-Amélioration
```bash
GET /api/improvement
# Réponse: {health_score, metrics, issues, suggestions, ...}
```

### Health Check
```bash
GET /api/health
# Réponse: {status: 'healthy'}
```

---

## 👁️ Interface

### Vue Principale

1. **Header**
   - Status en ligne (pulsant)
   - Dernière mise à jour

2. **Stats Cards**
   - Portfolio Value + variation
   - Cash disponible
   - Win Rate
   - Health Score avec barre de progression

3. **Graphiques**
   - Évolution portfolio (Chart.js)
   - Métriques de performance (Sharpe, Drawdown, etc.)

4. **Listes**
   - Positions actuelles avec P&L
   - 10 derniers trades

5. **Auto-Amélioration**
   - Problèmes détectés (avec sévérité)
   - Suggestions d'amélioration

### Rafraîchissement

- **Automatique** : Toutes les 10 secondes
- **Manuel** : Rechargez la page

---

## 🔒 Sécurité

### Recommandations

1. **VPN uniquement** : Ne pas exposer publiquement
2. **Firewall** : Bloquer port 5000 sauf VPN
3. **Auth (TODO)** : Ajouter authentification si nécessaire

### Configuration Firewall (UFW)

```bash
# Autoriser seulement depuis VPN (ex: 10.8.0.0/24)
sudo ufw allow from 10.8.0.0/24 to any port 5000

# Ou autoriser localement uniquement
# (utiliser tunnel SSH pour accès distant)
sudo ufw deny 5000
```

---

## 🐛 Troubleshooting

### Dashboard ne démarre pas

```bash
# Vérifier logs
sudo journalctl -u ploutos-dashboard -f

# Vérifier port
sudo netstat -tulpn | grep 5000

# Tester manuellement
cd /root/ploutos/project_ploutos/web
python3 app.py
```

### Données manquantes

```bash
# Vérifier logs trades
ls -lh /root/ploutos/project_ploutos/logs/trades/

# Vérifier bot actif
sudo systemctl status ploutos-trader-v2

# Vérifier Alpaca
python3 -c "from trading.alpaca_client import AlpacaClient; c = AlpacaClient(); print(c.get_account())"
```

### Erreur 503 (Service Unavailable)

Signifie qu'Alpaca ou Self-Improvement n'est pas disponible.

```bash
# Vérifier .env
cat .env | grep ALPACA

# Tester Alpaca
cd /root/ploutos/project_ploutos
python3 -c "from trading.alpaca_client import AlpacaClient; AlpacaClient()"

# Tester Self-Improvement
python3 core/self_improvement.py
```

---

## 🔧 Développement

### Ajouter une nouvelle route

```python
# Dans web/app.py
@app.route('/api/my_endpoint')
def my_endpoint():
    return jsonify({'data': 'value'})
```

### Modifier le frontend

```bash
# Éditer le template
nano web/templates/index.html

# Pas besoin de redémarrer Flask en mode debug
```

### Mode Debug

```bash
# Dans .env
DASHBOARD_DEBUG=true

# Redémarrer
sudo systemctl restart ploutos-dashboard
```

---

## 📦 TODO / Roadmap

- [ ] Authentification (login/password)
- [ ] Historique portfolio complet (depuis logs)
- [ ] Export PDF des rapports
- [ ] Notifications push
- [ ] Mode mobile responsive (déjà Tailwind)
- [ ] Graphiques supplémentaires (heatmap, correlation)
- [ ] Contrôle du bot (start/stop/restart)
- [ ] Logs en temps réel (WebSocket)

---

## 📝 License

Part of Ploutos AI Trading System - Private Use Only

---

## ❓ Support

Problème ? Ouvre une issue ou contacte l'équipe.

**Happy Trading!** 🚀
