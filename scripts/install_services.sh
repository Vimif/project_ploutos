#!/bin/bash
# install_services.sh

echo "🚀 INSTALLATION DES SERVICES 24/7"
echo "=================================="

# Vérifier qu'on est root
if [ "$EUID" -ne 0 ]; then 
   echo "❌ Ce script doit être lancé en root"
   exit 1
fi

# Chemins
PROJECT_DIR="/root/ploutos/project_ploutos"
VENV_DIR="/root/ploutos/venv"

# Vérifier que le projet existe
if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ Projet introuvable: $PROJECT_DIR"
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Venv introuvable: $VENV_DIR"
    exit 1
fi

echo "✅ Projet trouvé: $PROJECT_DIR"
echo "✅ Venv trouvé: $VENV_DIR"

# Créer dossier logs
mkdir -p "$PROJECT_DIR/data/logs"

# 1. Service Trading Bot
echo ""
echo "📝 Création du service ploutos-trader..."

cat > /etc/systemd/system/ploutos-trader.service << 'EOF'
[Unit]
Description=Ploutos Trading Bot - 24/7 Autonomous Trader
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/ploutos/project_ploutos
Environment="PYTHONPATH=/root/ploutos/project_ploutos"
Environment="PYTHONUNBUFFERED=1"
ExecStartPre=/bin/bash -c 'cd /root/ploutos/project_ploutos && git pull || true'
ExecStart=/root/ploutos/venv/bin/python scripts/run_trader.py --paper --interval 60 --capital 100000
Restart=always
RestartSec=30
TimeoutStartSec=300
TimeoutStopSec=30
StandardOutput=append:/root/ploutos/project_ploutos/data/logs/trader-service.log
StandardError=append:/root/ploutos/project_ploutos/data/logs/trader-service-error.log

[Install]
WantedBy=multi-user.target
EOF

# 2. Service Dashboard
echo "📝 Création du service ploutos-dashboard..."

cat > /etc/systemd/system/ploutos-dashboard.service << 'EOF'
[Unit]
Description=Ploutos Dashboard - Streamlit Interface
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/ploutos/project_ploutos
Environment="PYTHONPATH=/root/ploutos/project_ploutos"
Environment="PYTHONUNBUFFERED=1"
ExecStartPre=/bin/bash -c 'cd /root/ploutos/project_ploutos && git pull || true'
ExecStart=/root/ploutos/venv/bin/streamlit run ui/dashboard.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
Restart=always
RestartSec=30
TimeoutStartSec=60
TimeoutStopSec=10
StandardOutput=append:/root/ploutos/project_ploutos/data/logs/dashboard-service.log
StandardError=append:/root/ploutos/project_ploutos/data/logs/dashboard-service-error.log

[Install]
WantedBy=multi-user.target
EOF

# 3. Recharger systemd
echo ""
echo "🔄 Rechargement de systemd..."
systemctl daemon-reload

# 4. Activer les services
echo "✅ Activation du démarrage automatique..."
systemctl enable ploutos-trader.service
systemctl enable ploutos-dashboard.service

echo ""
echo "=================================="
echo "✅ INSTALLATION TERMINÉE"
echo "=================================="
echo ""
echo "📋 COMMANDES DISPONIBLES:"
echo ""
echo "# Démarrer les services"
echo "  systemctl start ploutos-trader"
echo "  systemctl start ploutos-dashboard"
echo ""
echo "# Arrêter les services"
echo "  systemctl stop ploutos-trader"
echo "  systemctl stop ploutos-dashboard"
echo ""
echo "# Voir le statut"
echo "  systemctl status ploutos-trader"
echo "  systemctl status ploutos-dashboard"
echo ""
echo "# Voir les logs en temps réel"
echo "  journalctl -u ploutos-trader -f"
echo "  journalctl -u ploutos-dashboard -f"
echo ""
echo "🌐 Dashboard accessible sur: http://$(hostname -I | awk '{print $1}'):8501"
echo ""
