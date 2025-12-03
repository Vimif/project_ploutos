#!/bin/bash
# scripts/setup_grafana.sh - Installation et configuration Grafana

set -e

echo "=================================="
echo "📊 Installation Grafana + Prometheus"
echo "=================================="

# Installer Prometheus
echo ""
echo "1️⃣ Installation de Prometheus..."
sudo apt-get update
sudo apt-get install -y prometheus

# Créer config Prometheus pour Ploutos
sudo tee /etc/prometheus/prometheus.yml > /dev/null << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ploutos_trading'
    static_configs:
      - targets: ['localhost:9090']
        labels:
          service: 'trading_bot'
          environment: 'production'
EOF

echo "✅ Prometheus configuré"

# Redémarrer Prometheus
sudo systemctl restart prometheus
sudo systemctl enable prometheus
echo "✅ Prometheus démarré"

# Installer Grafana
echo ""
echo "2️⃣ Installation de Grafana..."

# Ajouter repo Grafana
sudo apt-get install -y apt-transport-https software-properties-common wget
sudo mkdir -p /etc/apt/keyrings/
wget -q -O - https://apt.grafana.com/gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/grafana.gpg > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" | sudo tee /etc/apt/sources.list.d/grafana.list

# Installer Grafana
sudo apt-get update
sudo apt-get install -y grafana

echo "✅ Grafana installé"

# Démarrer Grafana
sudo systemctl daemon-reload
sudo systemctl start grafana-server
sudo systemctl enable grafana-server

echo "✅ Grafana démarré"

# Créer datasource Prometheus automatiquement
echo ""
echo "3️⃣ Configuration de la datasource Prometheus..."

# Attendre que Grafana démarre
sleep 5

# Créer datasource via API
curl -X POST http://admin:admin@localhost:3000/api/datasources \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Prometheus",
    "type": "prometheus",
    "url": "http://localhost:9090",
    "access": "proxy",
    "isDefault": true
  }' 2>/dev/null || echo "⚠️  Datasource déjà configurée ou erreur"

echo ""
echo "=================================="
echo "✅ INSTALLATION TERMINÉE"
echo "=================================="
echo ""
echo "📊 Grafana: http://localhost:3000"
echo "   Username: admin"
echo "   Password: admin (à changer au 1er login)"
echo ""
echo "📈 Prometheus: http://localhost:9090"
echo ""
echo "🔥 Métriques Ploutos: http://localhost:9090/metrics"
echo ""
echo "⏭️  PROCHAINE ÉTAPE:"
echo "   python scripts/import_grafana_dashboard.py"
echo ""