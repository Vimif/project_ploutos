# 🚀 Démarrage Rapide - Monitoring

## Installation en 5 minutes

### 1️⃣ Installer Grafana + Prometheus

cd /root/ploutos/project_ploutos
bash scripts/setup_grafana.sh

text
**⏱️ Temps : 2-3 minutes**

### 2️⃣ Importer le dashboard

python scripts/import_grafana_dashboard.py

text
**⏱️ Temps : 30 secondes**

### 3️⃣ Tester le système

python scripts/test_full_system.py

text
**⏱️ Temps : 1 minute**

### 4️⃣ Démarrer le bot

python -c "
from trading.live_trader import LiveTrader
trader = LiveTrader(paper_trading=True, monitoring_port=9090)
Les métriques sont maintenant actives!

"

text

### 5️⃣ Accéder au dashboard

🌐 http://localhost:3000
👤 Username: admin
🔑 Password: admin

text

---

## ✅ Checklist de vérification

- [ ] Prometheus accessible : http://localhost:9090
- [ ] Métriques visibles : http://localhost:9090/metrics
- [ ] Grafana accessible : http://localhost:3000
- [ ] Dashboard importé : "Ploutos Trading Bot"
- [ ] Alertes configurées (optionnel)

---

## 🎯 Utilisation

### Lancer le bot avec monitoring

from trading.live_trader import LiveTrader

trader = LiveTrader(
paper_trading=True,
monitoring_port=9090 # Port Prometheus
)

trader.run(check_interval_minutes=60)

text

### Vérifier les métriques

curl http://localhost:9090/metrics | grep ploutos

text

### Visualiser dans Grafana
1. Ouvrir http://localhost:3000
2. Aller dans Dashboards
3. Sélectionner "Ploutos Trading Bot - Live Monitoring"

---

## 📊 Ce que vous voyez

- **Portfolio en temps réel** : Valeur, cash, positions
- **P&L quotidien** : Gains/pertes du jour
- **Performance** : Win rate, Sharpe ratio
- **Risques** : Circuit breaker, positions à risque
- **Trades** : Historique et latence
- **Erreurs** : Monitoring des erreurs système