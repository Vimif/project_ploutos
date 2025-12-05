#!/usr/bin/env python3
"""Test du data fetcher corrigé"""
import sys
sys.path.insert(0, '.')

from core.data_fetcher import UniversalDataFetcher

print("\n🧪 TEST DU DATA FETCHER\n")

# Test 1 : Initialisation
fetcher = UniversalDataFetcher()

# Test 2 : Fetch 1 ticker (SPY)
print("\n" + "="*80)
print("Test 1 : Fetch SPY (ETF simple)")
print("="*80)

try:
    df_spy = fetcher.fetch('SPY', interval='1h')
    print(f"\n✅ SUCCESS : {len(df_spy)} bougies récupérées")
    print(f"\nPremières lignes :")
    print(df_spy.head())
    print(f"\nDernières lignes :")
    print(df_spy.tail())
except Exception as e:
    print(f"\n❌ ÉCHEC : {e}")

# Test 3 : Fetch action (NVDA)
print("\n" + "="*80)
print("Test 2 : Fetch NVDA (action)")
print("="*80)

try:
    df_nvda = fetcher.fetch('NVDA', interval='1h')
    print(f"\n✅ SUCCESS : {len(df_nvda)} bougies récupérées")
    print(df_nvda.tail())
except Exception as e:
    print(f"\n❌ ÉCHEC : {e}")

print("\n" + "="*80)
print("✅ Tests terminés")
print("="*80)
