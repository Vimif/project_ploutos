#!/bin/bash
# scripts/fix_dashboard_routes.sh
# Script de correction rapide pour les routes manquantes

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🔧 FIX ROUTES DASHBOARD${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Vérifier qu'on est dans le bon répertoire
if [ ! -f "dashboard/app.py" ]; then
    echo -e "${RED}❌ Erreur: Lancer depuis la racine du projet${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Répertoire correct${NC}"

# Ajouter les routes manquantes dans app.py
echo -e "${YELLOW}🔄 Ajout des routes /trades et /metrics...${NC}"

# Vérifier si les routes existent déjà
if grep -q "@app.route('/trades')" dashboard/app.py; then
    echo -e "${GREEN}✅ Routes déjà présentes${NC}"
else
    # Ajouter les routes avant la ligne "if __name__ == '__main__':"
    sed -i "/if __name__ == '__main__':/i\
# ========== ROUTES PAGES HTML ==========\n\n@app.route('/trades')\ndef trades_page():\n    \"\"\"Page des trades\"\"\"\n    return render_template('trades.html')\n\n@app.route('/metrics')\ndef metrics_page():\n    \"\"\"Page des métriques\"\"\"\n    return render_template('metrics.html')\n\n" dashboard/app.py
    
    echo -e "${GREEN}✅ Routes ajoutées${NC}"
fi

# Vérifier que les templates existent
if [ -f "dashboard/templates/trades.html" ] && [ -f "dashboard/templates/metrics.html" ]; then
    echo -e "${GREEN}✅ Templates présents${NC}"
else
    echo -e "${YELLOW}⚠️  Templates manquants, utiliser git pull${NC}"
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
echo -e "${GREEN}Accéder au dashboard:${NC}"
echo -e "  - Trades: http://localhost:5000/trades"
echo -e "  - Metrics: http://localhost:5000/metrics"
echo ""
