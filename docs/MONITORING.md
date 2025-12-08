# 📊 Monitoring Ploutos Trading

## 🎯 Vue d'ensemble

Système de monitoring complet avec Prometheus + Grafana pour surveillance en temps réel.

---

## 🚀 Installation

### 1. Installer Prometheus + Grafana

cd /root/ploutos/project_ploutos
bash scripts/setup_grafana.sh

text

**Installation complète :**
- ✅ Prometheus (collecte métriques)
- ✅ Grafana (visualisation)
- ✅ Configuration automatique
- ✅ Démarrage automatique

### 2. Importer le dashboard

python scripts/import_grafana_dashboard.py

text

### 3. Accéder à Grafana

URL: http://localhost:3000
Username: admin
Password: admin (à changer)

text

---

## 📈 Métriques disponibles

### Portfolio
- `ploutos_portfolio_value_usd` - Valeur totale
- `ploutos_cash_available_usd` - Cash disponible
- `ploutos_buying_power_usd` - Buying power
- `ploutos_positions_count` - Nombre de positions
- `ploutos_exposure_percent` - Exposition %

### Performance
- `ploutos_daily_pnl_usd` - P&L quotidien $
- `ploutos_daily_pnl_percent` - P&L quotidien %
- `ploutos_total_pnl_usd` - P&L total
- `ploutos_unrealized_pnl_usd` - P&L non réalisé
- `ploutos_win_rate_percent` - Taux de réussite

### Trading
- `ploutos_trades_total` - Compteur trades
- `ploutos_trade_amount_usd` - Distribution montants
- `ploutos_trade_latency_seconds` - Latence exécution
- `ploutos_predictions_total` - Compteur prédictions

### Risk Management
- `ploutos_circuit_breaker_active` - État circuit breaker
- `ploutos_risky_positions_count` - Positions à risque
- `ploutos_max_drawdown_percent` - Drawdown max
- `ploutos_sharpe_ratio` - Sharpe ratio

### Système
- `ploutos_errors_total` - Compteur erreurs
- `ploutos_alerts_total` - Compteur alertes
- `ploutos_api_request_duration_seconds` - Latence API

---

## 🎨 Dashboard Grafana

### Panneaux principaux

1. **💰 Portfolio Value** - Évolution temps réel
2. **📊 Daily P&L** - Profit/Loss quotidien
3. **💼 Positions** - Nombre et exposition
4. **🎯 Win Rate** - Gauge taux de réussite
5. **🚨 Circuit Breaker** - État de sécurité
6. **📈 Trades Timeline** - Historique trades
7. **⚠️ Risk Metrics** - Positions à risque
8. **📉 Performance** - Sharpe, Drawdown

---

## 🔧 Configuration avancée

### Modifier la fréquence de rafraîchissement

Dans Grafana, en haut à droite :
- 5s, 10s, 30s, 1m, 5m, 15m, 30m

### Ajouter des alertes Grafana

1. Ouvrir un panneau
2. Alert tab
3. Create Alert
4. Définir conditions (ex: `portfolio_value < 95000`)
5. Ajouter notification channel

### Exporter les données

Via Prometheus API

curl 'http://localhost:9090/api/v1/query?query=ploutos_portfolio_value_usd'
CSV depuis Grafana

Dashboard → Panel → More → Export CSV

text

---

## 📊 Requêtes Prometheus utiles

### Portfolio actuel

ploutos_portfolio_value_usd

text

### P&L sur 24h

ploutos_daily_pnl_usd

text

### Taux de trades par heure

rate(ploutos_trades_total[1h]) * 3600

text

### Latence médiane trades

histogram_quantile(0.5, rate(ploutos_trade_latency_seconds_bucket[5m]))

text

### Win rate glissant 7 jours

avg_over_time(ploutos_win_rate_percent[7d])

text

---

## 🚨 Alertes recommandées

### Circuit Breaker

ploutos_circuit_breaker_active == 1

text
→ Alerte critique immédiate

### Perte quotidienne > 2%

ploutos_daily_pnl_percent < -2

text
→ Alerte warning

### Win rate < 50%

ploutos_win_rate_percent < 50

text
→ Alerte info

### Positions à risque > 3

ploutos_risky_positions_count > 3

text
→ Alerte warning

---

## 🔍 Troubleshooting

### Métriques non visibles

Vérifier serveur Prometheus

curl http://localhost:9090/metrics
Vérifier bot live_trader lancé

ps aux | grep live_trader
Vérifier logs

tail -f logs/live_trader.log

text

### Grafana ne démarre pas

sudo systemctl status grafana-server
sudo journalctl -u grafana-server -f

text

### Dashboard vide

1. Vérifier datasource : Grafana → Configuration → Data Sources
2. Vérifier Prometheus : http://localhost:9090
3. Vérifier bot actif avec métriques

---

## 📱 Accès distant

### Tunnel SSH

ssh -L 3000:localhost:3000 user@server

text
→ Accès via http://localhost:3000

### Reverse proxy Nginx

server {
listen 80;
server_name monitoring.ploutos.com;

text
location / {
    proxy_pass http://localhost:3000;
}

}

text

---

## 🎯 Best Practices

1. **Rétention données** : Prometheus garde 15j par défaut
2. **Refresh rate** : 30s recommandé (pas trop fréquent)
3. **Alertes** : Configurer pour événements critiques
4. **Snapshots** : Prendre régulièrement des snapshots dashboard
5. **Backup** : Sauvegarder `/var/lib/grafana`
