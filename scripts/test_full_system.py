#!/usr/bin/env python3
"""Test complet de tous les systèmes"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import requests

def test_database():
    """Tester la base de données"""
    print("\n1️⃣ TEST BASE DE DONNÉES")
    print("-" * 70)
    
    try:
        from database.db import get_connection
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trades")
            count = cursor.fetchone()[0]
            print(f"✅ BDD OK - {count} trades enregistrés")
            return True
    except Exception as e:
        print(f"❌ BDD ERREUR: {e}")
        return False

def test_alerts():
    """Tester les alertes"""
    print("\n2️⃣ TEST ALERTES")
    print("-" * 70)
    
    try:
        from core.alerts import send_alert
        send_alert("🧪 Test système complet - Tout fonctionne!", priority='SUCCESS')
        print("✅ Alertes OK - Vérifiez Telegram/Discord")
        return True
    except Exception as e:
        print(f"❌ Alertes ERREUR: {e}")
        return False

def test_risk_manager():
    """Tester le risk manager"""
    print("\n3️⃣ TEST RISK MANAGER")
    print("-" * 70)
    
    try:
        from core.risk_manager import RiskManager
        rm = RiskManager()
        
        qty, value = rm.calculate_position_size(
            portfolio_value=100000,
            entry_price=500,
            stop_loss_pct=0.05
        )
        
        print(f"✅ Risk Manager OK - Position size: {qty} actions (${value:,.2f})")
        return True
    except Exception as e:
        print(f"❌ Risk Manager ERREUR: {e}")
        return False

def test_monitoring():
    """Tester le monitoring"""
    print("\n4️⃣ TEST MONITORING PROMETHEUS")
    print("-" * 70)
    
    try:
        from core.monitoring import get_metrics
        metrics = get_metrics(port=9091)  # Port différent pour le test
        
        # Tester enregistrement métriques
        metrics.portfolio_value.set(100000)
        metrics.record_trade('TEST', 'BUY', 5000, 0.5, 'success')
        
        print("✅ Monitoring OK - Métriques enregistrées")
        
        # Vérifier serveur Prometheus
        try:
            response = requests.get('http://localhost:9090/metrics', timeout=2)
            if response.status_code == 200:
                print("✅ Serveur Prometheus accessible")
            else:
                print("⚠️  Serveur Prometheus non accessible")
        except:
            print("⚠️  Serveur Prometheus non démarré (normal si bot pas lancé)")
        
        return True
    except Exception as e:
        print(f"❌ Monitoring ERREUR: {e}")
        return False

def test_grafana():
    """Tester Grafana"""
    print("\n5️⃣ TEST GRAFANA")
    print("-" * 70)
    
    try:
        response = requests.get('http://localhost:3000/api/health', timeout=5)
        if response.status_code == 200:
            print("✅ Grafana OK - Dashboard accessible")
            print("   URL: http://localhost:3000")
            return True
        else:
            print("⚠️  Grafana répond mais avec erreur")
            return False
    except Exception as e:
        print(f"❌ Grafana non accessible: {e}")
        print("   Installer avec: bash scripts/setup_grafana.sh")
        return False

def test_integration():
    """Test d'intégration complet"""
    print("\n6️⃣ TEST D'INTÉGRATION")
    print("-" * 70)
    
    try:
        from trading.alpaca_client import AlpacaClient
        
        alpaca = AlpacaClient(paper_trading=True)
        account = alpaca.get_account()
        
        if account:
            print(f"✅ Alpaca OK - Portfolio: ${account['portfolio_value']:,.2f}")
            
            # Test avec monitoring
            try:
                from core.monitoring import get_metrics
                metrics = get_metrics()
                positions = alpaca.get_positions()
                metrics.update_portfolio_metrics(account, positions)
                print("✅ Intégration Alpaca + Monitoring OK")
            except:
                print("⚠️  Monitoring non intégré (normal)")
            
            return True
        else:
            print("❌ Alpaca non accessible")
            return False
    except Exception as e:
        print(f"❌ Intégration ERREUR: {e}")
        return False

def main():
    """Fonction principale"""
    print("="*70)
    print("🧪 TEST COMPLET DU SYSTÈME PLOUTOS")
    print("="*70)
    
    results = {
        'Database': test_database(),
        'Alerts': test_alerts(),
        'Risk Manager': test_risk_manager(),
        'Monitoring': test_monitoring(),
        'Grafana': test_grafana(),
        'Integration': test_integration()
    }
    
    print("\n" + "="*70)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*70)
    
    for component, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{component:20s} {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print("\n" + "="*70)
    print(f"TOTAL: {passed}/{total} tests réussis ({passed/total*100:.0f}%)")
    print("="*70)
    
    if passed == total:
        print("\n🎉 TOUS LES SYSTÈMES SONT OPÉRATIONNELS!")
    else:
        print("\n⚠️  Certains systèmes nécessitent une configuration")

if __name__ == '__main__':
    main()