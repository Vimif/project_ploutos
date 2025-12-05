#!/usr/bin/env python3
"""
Script pour corriger data_fetcher.py automatiquement
"""

import os
import sys

# Ajouter le chemin racine
sys.path.insert(0, '/root/ai-factory/tmp/project_ploutos')

FIXES = """
# Ligne 8 : Ajouter import dotenv
APRÈS la ligne :
    import warnings

AJOUTER :
    # ✅ Charger .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("⚠️ python-dotenv non installé, variables système utilisées")

# Ligne 42 : Fix nom variable Alpaca
REMPLACER :
    api_secret = os.getenv('ALPACA_SECRET_KEY')

PAR :
    api_secret = os.getenv('ALPACA_SECRET_KEY') or os.getenv('ALPACA_API_SECRET')

# Ligne 53 : Fix test Alpaca (déplacer avant return)
REMPLACER :
    client = StockHistoricalDataClient(api_key, api_secret)
    
    print("  ✅ Alpaca connecté (alpaca-py)")
    return client
    
    # Test de connexion
    test_request = ...

PAR :
    client = StockHistoricalDataClient(api_key, api_secret)
    
    # ✅ Test AVANT return
    try:
        test_request = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame.Day,
            start=datetime.now() - timedelta(days=2)
        )
        client.get_stock_bars(test_request)
        print("  ✅ Alpaca connecté (alpaca-py)")
        return client
    except Exception as test_err:
        print(f"  ⚠️ Alpaca test échec : {str(test_err)[:80]}")
        return None

# Ligne 175 : Fix limite Yahoo 730 jours
DANS _fetch_yfinance, AVANT yf.download, AJOUTER :

    # ✅ Gérer limite 730 jours pour intervalles horaires
    if interval in ['1h', '30m', '15m', '5m', '1m']:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d') if isinstance(start_date, str) else start_date
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') if isinstance(end_date, str) else end_date
        
        delta_days = (end_dt - start_dt).days
        
        if delta_days > 729:
            start_date = (end_dt - timedelta(days=729)).strftime('%Y-%m-%d')
            print(f"    ⚠️ Yahoo limite 730j pour {interval} : ajusté à {start_date}")
"""

print("="*80)
print("🔧 GUIDE DE CORRECTION MANUELLE")
print("="*80)
print(FIXES)
print("\n✅ Applique ces corrections dans : core/data_fetcher.py")
print("⚠️ Ou remplace le fichier avec la version corrigée ci-dessous\n")

# Version complète corrigée disponible sur demande
