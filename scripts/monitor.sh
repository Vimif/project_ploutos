#!/bin/bash
# monitor.sh - Vérifier que tout tourne

echo "📊 MONITORING PLOUTOS"
echo "===================="

# Fonction pour vérifier un service
check_service() {
    local service=$1
    if systemctl is-active --quiet $service; then
        echo "✅ $service: ACTIF"
        uptime=$(systemctl show -p ActiveEnterTimestamp $service | cut -d= -f2)
        echo "   Uptime: $uptime"
    else
        echo "❌ $service: ARRÊTÉ"
    fi
}

echo ""
echo "Services:"
check_service ploutos-trader
check_service ploutos-dashboard

echo ""
echo "Logs récents (trader):"
tail -n 5 /root/ploutos/project_ploutos/data/logs/trader-service.log 2>/dev/null || echo "   Pas de logs"

echo ""
echo "Logs récents (dashboard):"
tail -n 5 /root/ploutos/project_ploutos/data/logs/dashboard-service.log 2>/dev/null || echo "   Pas de logs"

echo ""
echo "Processus:"
ps aux | grep -E "(streamlit|run_trader)" | grep -v grep

echo ""
echo "Ports:"
ss -tlnp | grep -E "(8501)" || echo "   Dashboard non accessible"

echo ""
echo "Modèles disponibles:"
ls -lh /root/ploutos/project_ploutos/data/models/*.zip 2>/dev/null | wc -l || echo "0"
