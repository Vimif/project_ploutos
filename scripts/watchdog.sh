#!/bin/bash
# watchdog.sh - Relancer si crash

LOG_FILE="/root/ploutos/project_ploutos/data/logs/watchdog.log"

check_and_restart() {
    local service=$1
    
    if ! systemctl is-active --quiet $service; then
        echo "[$(date)] ⚠️  $service est arrêté, redémarrage..." >> $LOG_FILE
        systemctl restart $service
        sleep 10
        
        if systemctl is-active --quiet $service; then
            echo "[$(date)] ✅ $service redémarré avec succès" >> $LOG_FILE
        else
            echo "[$(date)] ❌ Échec du redémarrage de $service" >> $LOG_FILE
        fi
    fi
}

# Vérifier les deux services
check_and_restart ploutos-trader
check_and_restart ploutos-dashboard
