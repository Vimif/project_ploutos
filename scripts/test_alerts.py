#!/usr/bin/env python3
"""Tester le système d'alertes"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.alerts import (
    send_alert, alert_trade, alert_profit, alert_loss,
    alert_daily_summary, alert_startup
)

def test_all_alerts():
    """Tester toutes les alertes"""
    
    print("🔔 Test du système d'alertes Ploutos\n")
    
    # 1. Alerte démarrage
    print("1. Test alerte démarrage...")
    alert_startup()
    input("   Appuyez sur Entrée pour continuer...")
    
    # 2. Alerte trade
    print("2. Test alerte trade...")
    alert_trade('NVDA', 'BUY', 10, 500.50, 5005.00)
    input("   Appuyez sur Entrée pour continuer...")
    
    # 3. Alerte profit
    print("3. Test alerte profit...")
    alert_profit('MSFT', 1234.56, 8.5)
    input("   Appuyez sur Entrée pour continuer...")
    
    # 4. Alerte perte
    print("4. Test alerte perte...")
    alert_loss('TSLA', -567.89, -4.2)
    input("   Appuyez sur Entrée pour continuer...")
    
    # 5. Résumé quotidien
    print("5. Test résumé quotidien...")
    alert_daily_summary(112567.89, 2345.67, 2.12, 15)
    input("   Appuyez sur Entrée pour continuer...")
    
    # 6. Alerte custom
    print("6. Test alerte personnalisée...")
    send_alert(
        "✅ **Test réussi!**\n\n"
        "Tous les types d'alertes fonctionnent correctement.",
        priority='SUCCESS'
    )
    
    print("\n✅ Tests terminés!")

if __name__ == '__main__':
    test_all_alerts()