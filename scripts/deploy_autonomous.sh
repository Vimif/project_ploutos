#!/bin/bash

# Script de déploiement automatique du système autonome
# À lancer sur le VPS

set -e  # Exit on error

echo "============================================"
echo "🚀 DÉPLOIEMENT SYSTÈME AUTONOME"
echo "============================================"

# Variables
PROJECT_DIR="/root/ploutos/project_ploutos"
VENV_DIR="/root/ai-factory/venv"
PYTHON="$VENV_DIR/bin/python3"

cd $PROJECT_DIR

# 1. Installer dépendances
echo ""
echo "📦 Installation dépendances..."
$VENV_DIR/bin/pip install -q optuna pyyaml

# 2. Créer dossiers
echo "📁 Création structure..."
mkdir -p models/autonomous
mkdir -p models/backups
mkdir -p reports/autonomous
mkdir -p logs
mkdir -p data

# 3. Setup CRON pour apprentissage continu
echo ""
echo "⏰ Configuration CRON..."

# Créer le script CRON
cat > /tmp/ploutos_cron << 'EOF'
# Apprentissage continu tous les dimanches à 2h du matin
0 2 * * 0 /root/ai-factory/venv/bin/python3 /root/ploutos/project_ploutos/scripts/continuous_learning.py >> /root/ploutos/logs/continuous_learning.log 2>&1

# Backup hebdomadaire tous les lundis à 3h du matin
0 3 * * 1 tar -czf /root/ploutos/backups/backup_$(date +\%Y\%m\%d).tar.gz /root/ploutos/project_ploutos/models /root/ploutos/project_ploutos/data

# Nettoyage des vieux backups (>30 jours) tous les 1er du mois
0 4 1 * * find /root/ploutos/backups -name "backup_*.tar.gz" -mtime +30 -delete
EOF

# Installer dans crontab
crontab -l > /tmp/old_cron 2>/dev/null || true
cat /tmp/old_cron /tmp/ploutos_cron | sort -u | crontab -

echo "✅ CRON configuré"

# 4. Créer service systemd pour bot autonome
echo ""
echo "🔧 Configuration systemd..."

cat > /etc/systemd/system/ploutos-autonomous.service << EOF
[Unit]
Description=Ploutos Autonomous Trading Bot
After=network.target postgresql.service

[Service]
Type=simple
User=root
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$PYTHON $PROJECT_DIR/trading/autonomous_trader.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ploutos-autonomous.service

echo "✅ Service créé (pas démarré)"

# 5. Premier entraînement
echo ""
echo "🧠 Lancement premier entraînement..."
echo "   (Cela peut prendre plusieurs heures)"

$PYTHON scripts/autonomous_system.py --skip-optimization

# 6. Vérifier résultats
if [ -f "models/autonomous/production.zip" ]; then
    echo ""
    echo "============================================"
    echo "✅ DÉPLOIEMENT RÉUSSI"
    echo "============================================"
    echo ""
    echo "📊 Modèle : models/autonomous/production.zip"
    echo "⏰ CRON   : Apprentissage continu activé"
    echo "🤖 Service: ploutos-autonomous.service"
    echo ""
    echo "Pour démarrer le bot:"
    echo "  sudo systemctl start ploutos-autonomous"
    echo ""
    echo "Pour voir les logs:"
    echo "  tail -f logs/continuous_learning.log"
else
    echo ""
    echo "❌ DÉPLOIEMENT ÉCHOUÉ"
    echo "Vérifier les logs ci-dessus"
    exit 1
fi
