#!/bin/bash
# scripts/patch_dashboard_v2.sh
# Script de déploiement du patch Dashboard v2.0

set -e  # Exit si erreur

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================${NC}"
echo -e "${BLUE}🔧 PATCH DASHBOARD PLOUTOS V2.0${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""

# Vérifier qu'on est dans le bon répertoire
if [ ! -f "dashboard/app.py" ]; then
    echo -e "${RED}❌ Erreur: Lancer depuis la racine du projet${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Répertoire correct${NC}"

# Créer un backup
BACKUP_DIR="backups/dashboard_$(date +%Y%m%d_%H%M%S)"
echo -e "${YELLOW}📦 Création backup dans ${BACKUP_DIR}...${NC}"

mkdir -p "$BACKUP_DIR"
cp -r dashboard/app.py "$BACKUP_DIR/"
cp -r dashboard/requirements.txt "$BACKUP_DIR/" 2>/dev/null || true
cp -r dashboard/templates "$BACKUP_DIR/" 2>/dev/null || true

echo -e "${GREEN}✅ Backup créé${NC}"

# Vérifier que les nouveaux fichiers existent
if [ ! -f "dashboard/app_v2.py" ]; then
    echo -e "${RED}❌ Erreur: dashboard/app_v2.py introuvable${NC}"
    exit 1
fi

if [ ! -f "dashboard/analytics.py" ]; then
    echo -e "${RED}❌ Erreur: dashboard/analytics.py introuvable${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Nouveaux fichiers détectés${NC}"

# Sauvegarder l'ancien app.py en app_legacy.py
echo -e "${YELLOW}📝 Sauvegarde app.py -> app_legacy.py${NC}"
cp dashboard/app.py dashboard/app_legacy.py

# Copier le nouveau dashboard
echo -e "${YELLOW}🔄 Installation dashboard v2.0...${NC}"
cp dashboard/app_v2.py dashboard/app.py

echo -e "${GREEN}✅ Dashboard v2.0 installé${NC}"

# Mettre à jour requirements
echo -e "${YELLOW}📦 Mise à jour des dépendances...${NC}"

cat > dashboard/requirements_v2.txt << 'EOF'
# Requirements Dashboard v2.0
flask==3.0.0
flask-cors==4.0.0
flask-socketio==5.3.5
psycopg2-binary==2.9.9
gevent==23.9.1
numpy>=1.24.0
pandas>=2.0.0
EOF

echo -e "${GREEN}✅ Requirements v2 créés${NC}"

# Installer les dépendances si virtualenv actif
if [ -n "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}📦 Installation des dépendances...${NC}"
    pip install -q -r dashboard/requirements_v2.txt
    echo -e "${GREEN}✅ Dépendances installées${NC}"
else
    echo -e "${YELLOW}⚠️  Virtualenv non actif, skip installation${NC}"
    echo -e "${YELLOW}   Installer manuellement: pip install -r dashboard/requirements_v2.txt${NC}"
fi

# Vérifier PostgreSQL
echo -e "${YELLOW}🔍 Vérification PostgreSQL...${NC}"

if command -v psql &> /dev/null; then
    if psql -U ploutos -d ploutos -c "SELECT 1" &> /dev/null; then
        echo -e "${GREEN}✅ PostgreSQL accessible${NC}"
        PG_OK=true
    else
        echo -e "${YELLOW}⚠️  PostgreSQL non accessible (mot de passe?)${NC}"
        echo -e "${YELLOW}   Dashboard fonctionnera en mode JSON${NC}"
        PG_OK=false
    fi
else
    echo -e "${YELLOW}⚠️  psql non installé, impossible de vérifier${NC}"
    echo -e "${YELLOW}   Dashboard fonctionnera en mode JSON si BDD indisponible${NC}"
    PG_OK=false
fi

# Test import Python
echo -e "${YELLOW}🐍 Test des imports Python...${NC}"

python3 << 'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

try:
    from dashboard.analytics import PortfolioAnalytics
    print("✅ Module analytics OK")
except Exception as e:
    print(f"❌ Erreur import analytics: {e}")
    sys.exit(1)

try:
    import numpy as np
    import pandas as pd
    print("✅ numpy/pandas OK")
except Exception as e:
    print(f"❌ Erreur numpy/pandas: {e}")
    sys.exit(1)

PYEOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Imports Python OK${NC}"
else
    echo -e "${RED}❌ Erreur imports Python${NC}"
    echo -e "${YELLOW}Restauration du backup...${NC}"
    cp "$BACKUP_DIR/app.py" dashboard/app.py
    echo -e "${RED}❌ Patch annulé${NC}"
    exit 1
fi

# Redémarrer le service si actif
if systemctl is-active --quiet ploutos-trader-v2.service 2>/dev/null; then
    echo -e "${YELLOW}🔄 Redémarrage du service...${NC}"
    sudo systemctl restart ploutos-trader-v2.service
    sleep 2
    
    if systemctl is-active --quiet ploutos-trader-v2.service; then
        echo -e "${GREEN}✅ Service redémarré${NC}"
    else
        echo -e "${RED}❌ Erreur redémarrage service${NC}"
        echo -e "${YELLOW}Vérifier les logs: journalctl -u ploutos-trader-v2.service -f${NC}"
    fi
else
    echo -e "${YELLOW}ℹ️  Service non actif, pas de redémarrage${NC}"
fi

echo ""
echo -e "${BLUE}=================================================${NC}"
echo -e "${GREEN}✅ PATCH DASHBOARD V2.0 INSTALLÉ${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""
echo -e "${GREEN}📊 Nouveautés:${NC}"
echo -e "   - Métriques avancées (Sharpe, Sortino, Calmar)"
echo -e "   - Analyse drawdown et risque"
echo -e "   - Analytics par symbole"
echo -e "   - Connexion PostgreSQL avec fallback JSON"
echo ""
echo -e "${GREEN}📝 Fichiers:${NC}"
echo -e "   - dashboard/app.py (nouveau v2.0)"
echo -e "   - dashboard/app_legacy.py (ancien backup)"
echo -e "   - dashboard/analytics.py (module analytics)"
echo -e "   - Backup complet: ${BACKUP_DIR}"
echo ""
echo -e "${GREEN}🚀 Démarrage:${NC}"
echo -e "   cd dashboard && python app.py"
echo ""
echo -e "${GREEN}🔗 Endpoints nouveaux:${NC}"
echo -e "   GET /api/analytics/advanced?days=30"
echo -e "   GET /api/analytics/symbol/<SYMBOL>?days=30"
echo -e "   GET /api/health"
echo ""
echo -e "${YELLOW}⚠️  En cas de problème:${NC}"
echo -e "   cp dashboard/app_legacy.py dashboard/app.py"
echo ""
