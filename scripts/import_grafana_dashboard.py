#!/usr/bin/env python3
"""Importer automatiquement le dashboard Grafana"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time

import requests


def wait_for_grafana(max_attempts=30):
    """Attendre que Grafana soit prêt"""
    print("⏳ Attente démarrage Grafana...")

    for i in range(max_attempts):
        try:
            response = requests.get('http://localhost:3000/api/health', timeout=2)
            if response.status_code == 200:
                print("✅ Grafana est prêt!")
                return True
        except Exception:
            pass

        time.sleep(2)
        print(f"   Tentative {i+1}/{max_attempts}...")

    return False

def create_datasource():
    """Créer la datasource Prometheus"""
    print("\n📊 Configuration datasource Prometheus...")

    datasource = {
        "name": "Prometheus",
        "type": "prometheus",
        "url": "http://localhost:9090",
        "access": "proxy",
        "isDefault": True,
        "jsonData": {
            "httpMethod": "POST"
        }
    }

    try:
        response = requests.post(
            'http://admin:admin@localhost:3000/api/datasources',
            json=datasource,
            headers={'Content-Type': 'application/json'}
        )

        if response.status_code in [200, 409]:  # 409 = déjà existe
            print("✅ Datasource Prometheus configurée")
            return True
        else:
            print(f"⚠️  Erreur datasource: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def import_dashboard():
    """Importer le dashboard"""
    print("\n📊 Import du dashboard...")

    dashboard_path = Path(__file__).parent.parent / 'config' / 'grafana_dashboard.json'

    if not dashboard_path.exists():
        print(f"❌ Dashboard non trouvé: {dashboard_path}")
        return False

    with open(dashboard_path) as f:
        dashboard_json = json.load(f)

    try:
        response = requests.post(
            'http://admin:admin@localhost:3000/api/dashboards/db',
            json=dashboard_json,
            headers={'Content-Type': 'application/json'}
        )

        if response.status_code == 200:
            result = response.json()
            dashboard_url = f"http://localhost:3000{result.get('url', '')}"
            print("✅ Dashboard importé avec succès!")
            print(f"🔗 URL: {dashboard_url}")
            return True
        else:
            print(f"⚠️  Erreur import: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    """Fonction principale"""
    print("="*70)
    print("📊 IMPORT DASHBOARD GRAFANA PLOUTOS")
    print("="*70)

    # Attendre Grafana
    if not wait_for_grafana():
        print("❌ Grafana ne démarre pas. Vérifier les logs:")
        print("   sudo systemctl status grafana-server")
        return

    # Créer datasource
    time.sleep(2)
    create_datasource()

    # Importer dashboard
    time.sleep(2)
    if import_dashboard():
        print("\n" + "="*70)
        print("✅ CONFIGURATION TERMINÉE")
        print("="*70)
        print("\n📊 Accédez à Grafana:")
        print("   URL: http://localhost:3000")
        print("   Username: admin")
        print("   Password: admin")
        print("\n🔥 Dashboard: Ploutos Trading Bot - Live Monitoring")
        print("\n⚠️  Changez le mot de passe admin au premier login!")
        print("="*70)
    else:
        print("\n❌ Échec import dashboard")
        print("   Vous pouvez l'importer manuellement depuis:")
        print(f"   {Path(__file__).parent.parent / 'config' / 'grafana_dashboard.json'}")

if __name__ == '__main__':
    main()
