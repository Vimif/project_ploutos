#!/bin/bash
# scripts/fix_homepage.sh
# Script de correction rapide pour la page d'accueil

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🔧 FIX PAGE D'ACCUEIL${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ ! -f "dashboard/app.py" ]; then
    echo -e "${RED}❌ Erreur: Lancer depuis la racine du projet${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Répertoire correct${NC}"

# Backup ancien template
if [ -f "dashboard/templates/index.html" ]; then
    cp dashboard/templates/index.html dashboard/templates/index_old.html
    echo -e "${YELLOW}💾 Backup: index_old.html${NC}"
fi

# Remplacer par le nouveau template
if [ -f "dashboard/templates/index_v2.html" ]; then
    cp dashboard/templates/index_v2.html dashboard/templates/index.html
    echo -e "${GREEN}✅ Template index.html mis à jour${NC}"
else
    echo -e "${YELLOW}⚠️  index_v2.html non trouvé, utiliser git pull${NC}"
fi

# Redémarrer le service si actif
if systemctl is-active --quiet ploutos-trader-v2.service 2>/dev/null; then
    echo -e "${YELLOW}🔄 Redémarrage du service...${NC}"
    sudo systemctl restart ploutos-trader-v2.service
    sleep 2
    
    if systemctl is-active --quiet ploutos-trader-v2.service; then
        echo -e "${GREEN}✅ Service redémarré${NC}"
    else
        echo -e "${RED}❌ Erreur redémarrage${NC}"
    fi
else
    echo -e "${YELLOW}ℹ️  Service non actif${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ FIX TERMINÉ${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Rafraîchir le navigateur (Ctrl+Shift+R)${NC}"
echo ""
