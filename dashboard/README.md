# 🏛️ PLOUTOS DASHBOARD V4

Interface web moderne pour visualiser le bot de trading en temps réel.

---

## ✨ FEATURES

✅ **Dashboard Temps Réel**
- Portfolio actuel (valeur + return)
- Positions ouvertes avec PnL
- Trades du jour
- Prédictions modèle

✅ **API REST Complète**
- `/api/status` : Status bot + portfolio
- `/api/trades` : Historique trades avec filtres
- `/api/metrics` : Métriques performance
- `/api/predictions` : Prédictions modèle

✅ **Design Moderne**
- Dark theme
- Responsive
- Auto-refresh (5s)
- Chart.js graphs

---

## 🚀 INSTALLATION

### **1. Dépendances**

```bash
cd dashboard
pip install -r requirements.txt
```

**Requirements** :
- Flask 3.0.0
- Flask-CORS 4.0.0  
- psycopg2-binary 2.9.9
- numpy 1.26.2
- gunicorn 21.2.0 (production)

---

### **2. Configuration PostgreSQL**

**Variables d'environnement** :

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=ploutos
export DB_USER=ploutos
export DB_PASSWORD=your_password
```

**Ou créer `.env`** :

```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ploutos
DB_USER=ploutos
DB_PASSWORD=your_password
PORT=5000
DEBUG=True
```

---

### **3. Structure Base de Données**

Le dashboard s'attend à ces tables :

**`trades`** :
```sql
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    action VARCHAR(10),
    shares INT,
    entry_price FLOAT,
    exit_price FLOAT,
    entry_time TIMESTAMP,
    exit_time TIMESTAMP,
    pnl FLOAT,
    pnl_pct FLOAT,
    commission FLOAT,
    current_price FLOAT
);
```

**`daily_summary`** :
```sql
CREATE TABLE daily_summary (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    portfolio_value FLOAT,
    balance FLOAT,
    daily_return FLOAT,
    n_trades INT
);
```

**`predictions`** :
```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    action VARCHAR(10),
    confidence FLOAT,
    timestamp TIMESTAMP,
    features JSONB
);
```

---

## 🏃 LANCEMENT

### **Mode Développement** (Local)

```bash
python3 app.py
```

**Accès** :
- Dashboard : http://localhost:5000
- API : http://localhost:5000/api/status

---

### **Mode Production** (Gunicorn)

```bash
gunicorn -w 4 -b 0.0.0.0:8080 app:app
```

**Options** :
- `-w 4` : 4 workers
- `-b 0.0.0.0:8080` : Bind port 8080
- `--daemon` : Run en background

**Avec logs** :
```bash
gunicorn -w 4 -b 0.0.0.0:8080 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log \
  app:app
```

---

### **Service Systemd** (VPS)

Créer `/etc/systemd/system/ploutos-dashboard.service` :

```ini
[Unit]
Description=Ploutos Dashboard
After=network.target postgresql.service

[Service]
User=root
WorkingDirectory=/root/ploutos/project_ploutos/dashboard
Environment="DB_HOST=localhost"
Environment="DB_NAME=ploutos"
Environment="DB_USER=ploutos"
Environment="DB_PASSWORD=your_password"
ExecStart=/root/ploutos/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

**Activer** :
```bash
sudo systemctl daemon-reload
sudo systemctl enable ploutos-dashboard
sudo systemctl start ploutos-dashboard
sudo systemctl status ploutos-dashboard
```

---

## 📊 API ENDPOINTS

### **GET /api/status**

Status bot + portfolio actuel

**Response** :
```json
{
  "success": true,
  "timestamp": "2025-12-09T22:00:00",
  "portfolio": {
    "balance": 50000.00,
    "portfolio_value": 125000.00,
    "return_pct": 25.0
  },
  "trades_today": 12,
  "positions": [
    {
      "ticker": "NVDA",
      "shares": 50,
      "entry_price": 450.00,
      "current_price": 475.00,
      "pnl_pct": 5.55
    }
  ],
  "recent_predictions": []
}
```

---

### **GET /api/trades**

Historique trades avec filtres

**Params** :
- `days` : Jours historique (défaut 30)
- `ticker` : Filtrer par ticker
- `limit` : Limite résultats (défaut 100)

**Exemple** :
```bash
curl "http://localhost:5000/api/trades?days=90&ticker=NVDA&limit=50"
```

**Response** :
```json
{
  "success": true,
  "trades": [
    {
      "id": 1234,
      "ticker": "NVDA",
      "action": "BUY",
      "shares": 50,
      "entry_price": 450.00,
      "exit_price": 475.00,
      "pnl_pct": 5.55,
      "entry_time": "2025-12-01T10:30:00",
      "exit_time": "2025-12-09T15:45:00"
    }
  ],
  "stats": {
    "total_trades": 245,
    "winning_trades": 142,
    "avg_pnl_pct": 2.3,
    "max_win": 15.2,
    "max_loss": -4.8
  }
}
```

---

### **GET /api/metrics**

Métriques performance

**Params** :
- `days` : Période (défaut 90)

**Response** :
```json
{
  "success": true,
  "metrics": {
    "total_return": 25.0,
    "sharpe_ratio": 2.1,
    "max_drawdown": 5.2,
    "current_value": 125000.00,
    "total_trades": 245,
    "avg_trades_per_day": 18.5
  },
  "daily_data": []
}
```

---

### **GET /api/health**

Healthcheck

**Response** :
```json
{
  "success": true,
  "status": "healthy",
  "database": "ok",
  "timestamp": "2025-12-09T22:00:00"
}
```

---

## 🛠️ TROUBLESHOOTING

### **"Can't connect to database"**

```bash
# Vérifier PostgreSQL
sudo systemctl status postgresql

# Vérifier connexion
psql -U ploutos -d ploutos -h localhost

# Vérifier variables env
env | grep DB_
```

---

### **"Port 5000 already in use"**

```bash
# Changer port
export PORT=8080
python3 app.py

# Ou tuer process
sudo lsof -ti:5000 | xargs kill -9
```

---

### **"Module not found"**

```bash
# Installer dépendances
pip install -r requirements.txt

# Vérifier virtualenv
which python3
```

---

## 📸 SCREENSHOTS

### **Dashboard Principal**
- Portfolio en temps réel
- Positions ouvertes
- Prédictions modèle

### **Page Trades**
- Historique complet
- Filtres par ticker/période
- Stats performance

### **Page Métriques**
- Graphique portfolio
- Sharpe ratio
- Drawdown
- Win rate

---

## 🔒 SÉCURITÉ

⚠️ **En production** :

1. **Changer password DB** dans `.env`
2. **Utiliser HTTPS** (Nginx reverse proxy)
3. **Restreindre accès** (VPN ou firewall)
4. **Désactiver DEBUG** :
   ```bash
   export DEBUG=False
   ```

---

## 📝 TODO

- [ ] Authentification (login/password)
- [ ] WebSocket pour updates temps réel
- [ ] Graphiques avancés (Plotly)
- [ ] Export CSV trades
- [ ] Notifications Discord/Telegram
- [ ] Mode mobile optimisé

---

**Date** : 9 Décembre 2025  
**Version** : V4  
**Auteur** : Ploutos AI Team  
**Status** : ✅ PRÊT À UTILISER
