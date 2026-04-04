#!/bin/bash
# scripts/setup_grafana_v2.sh - Version corrigée

set -e

echo "=================================="
echo "📊 Installation Grafana + Prometheus (FIXED)"
echo "=================================="

# 1. INSTALLATION PROMETHEUS (D'ABORD !)
echo ""
echo "1️⃣ Installation de Prometheus..."
sudo apt-get update
# On force la config par défaut pour éviter les questions interactives
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y prometheus prometheus-node-exporter

echo "✅ Prometheus installé"

# 2. CONFIGURATION PROMETHEUS (MAINTENANT ON ÉCRASE)
echo ""
echo "2️⃣ Application de la configuration..."
sudo systemctl stop prometheus

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

# Redémarrer avec la nouvelle config
sudo systemctl restart prometheus
sudo systemctl enable prometheus
echo "✅ Prometheus configuré et redémarré"

# 3. INSTALLATION GRAFANA
echo ""
echo "3️⃣ Installation de Grafana..."

# Prérequis
sudo apt-get install -y apt-transport-https software-properties-common wget

# Clé et Repo
sudo mkdir -p /etc/apt/keyrings/
wget -q -O - https://apt.grafana.com/gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/grafana.gpg > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" | sudo tee /etc/apt/sources.list.d/grafana.list

# Install
sudo apt-get update
sudo apt-get install -y grafana

# Démarrage
sudo systemctl daemon-reload
sudo systemctl start grafana-server
sudo systemctl enable grafana-server

echo "✅ Grafana démarré"

# 4. CONFIGURATION DATASOURCE
echo ""
echo "4️⃣ Connexion Grafana -> Prometheus..."
sleep 10  # Attendre un peu plus que Grafana soit prêt

GRAFANA_URL=${GRAFANA_URL:-"http://localhost:3000"}
GRAFANA_USER=${GRAFANA_USER:-"admin"}
GRAFANA_PASSWORD=${GRAFANA_PASSWORD:-"admin"}

curl -X POST "${GRAFANA_URL}/api/datasources" \
  -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Prometheus",
    "type": "prometheus",
    "url": "http://localhost:9090",
    "access": "proxy",
    "isDefault": true
  }' 2>/dev/null || echo "⚠️  Datasource déjà configurée"

echo ""
echo "=================================="
echo "✅ INSTALLATION RÉUSSIE !"
echo "=================================="