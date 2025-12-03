#!/bin/bash
# Script de mise à jour automatique

cd /root/ploutos/project_ploutos

# Pull dernière version
git pull origin main

# Installer dépendances
source /root/ploutos/venv/bin/activate
pip install -r requirements.txt

# Redémarrer services
sudo systemctl restart ploutos-dashboard
sudo systemctl restart ploutos-trader

echo "✅ Mise à jour terminée"