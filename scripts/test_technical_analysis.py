#!/usr/bin/env python3
"""Script de test pour le module d'analyse technique

Vérifie que:
1. Le module technical_analysis s'importe correctement
2. Les indicateurs sont calculés sans erreur
3. Les signaux sont générés correctement
4. Aucune régression sur le code existant

Usage:
    python scripts/test_technical_analysis.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import traceback
from datetime import datetime

try:
    from dashboard.technical_analysis import TechnicalAnalyzer, TradingSignal
    print("✅ Import du module technical_analysis réussi")
except Exception as e:
    print(f"❌ Erreur import technical_analysis: {e}")
    sys.exit(1)


def test_basic_analysis():
    """Test analyse basique sur NVDA"""
    print("\n" + "="*70)
    print("🧪 TEST 1: Analyse basique NVDA")
    print("="*70)
    
    try:
        analyzer = TechnicalAnalyzer('NVDA', period='3mo', interval='1d')
        print(f"✅ Données téléchargées: {len(analyzer.df)} barres")
        
        # Test indicateurs individuels
        rsi = analyzer.calculate_rsi()
        print(f"✅ RSI calculé: {rsi.iloc[-1]:.2f}")
        
        macd_line, signal_line, histogram = analyzer.calculate_macd()
        print(f"✅ MACD calculé: {macd_line.iloc[-1]:.2f}")
        
        upper, middle, lower = analyzer.calculate_bollinger_bands()
        print(f"✅ Bollinger Bands calculées: {middle.iloc[-1]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test basique: {e}")
        traceback.print_exc()
        return False


def test_signal_generation():
    """Test génération de signal"""
    print("\n" + "="*70)
    print("🚦 TEST 2: Génération de signal")
    print("="*70)
    
    try:
        analyzer = TechnicalAnalyzer('AAPL', period='1mo', interval='1h')
        signal = analyzer.generate_signal()
        
        print(f"✅ Signal généré: {signal.signal}")
        print(f"   Strength: {signal.strength}/100")
        print(f"   Trend: {signal.trend}")
        print(f"   Confidence: {signal.confidence:.2%}")
        print(f"   Entry Price: ${signal.entry_price:.2f}")
        
        if signal.stop_loss:
            print(f"   Stop Loss: ${signal.stop_loss:.2f}")
        if signal.take_profit:
            print(f"   Take Profit: ${signal.take_profit:.2f}")
        
        print(f"\n   Raisons ({len(signal.reasons)}):")
        for i, reason in enumerate(signal.reasons[:5], 1):
            print(f"     {i}. {reason}")
        
        # Vérifications de sécurité
        assert signal.signal in ['BUY', 'SELL', 'HOLD'], "Signal invalide"
        assert 0 <= signal.strength <= 100, "Strength hors limites"
        assert 0.0 <= signal.confidence <= 1.0, "Confidence hors limites"
        assert signal.trend in ['BULLISH', 'BEARISH', 'NEUTRAL'], "Trend invalide"
        
        print("\n✅ Toutes les assertions passées")
        return True
        
    except Exception as e:
        print(f"❌ Erreur génération signal: {e}")
        traceback.print_exc()
        return False


def test_all_indicators():
    """Test récupération de tous les indicateurs"""
    print("\n" + "="*70)
    print("📊 TEST 3: Tous les indicateurs")
    print("="*70)
    
    try:
        analyzer = TechnicalAnalyzer('MSFT', period='6mo', interval='1d')
        indicators = analyzer.get_all_indicators()
        
        # Vérifier la structure
        assert 'price' in indicators, "Prix manquant"
        assert 'moving_averages' in indicators, "Moyennes mobiles manquantes"
        assert 'macd' in indicators, "MACD manquant"
        assert 'momentum' in indicators, "Momentum manquant"
        assert 'volatility' in indicators, "Volatilité manquante"
        assert 'volume' in indicators, "Volume manquant"
        
        print("✅ Structure des indicateurs valide")
        print(f"\n   Prix actuel: ${indicators['price']['current']:.2f}")
        print(f"   RSI: {indicators['momentum']['rsi']:.2f}")
        print(f"   SMA 20: ${indicators['moving_averages']['sma_20']:.2f}")
        print(f"   ATR: ${indicators['volatility']['atr']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur indicateurs: {e}")
        traceback.print_exc()
        return False


def test_multiple_symbols():
    """Test analyse multiple symboles"""
    print("\n" + "="*70)
    print("📋 TEST 4: Analyse multiple (batch)")
    print("="*70)
    
    symbols = ['NVDA', 'AAPL', 'MSFT']
    results = {}
    
    for symbol in symbols:
        try:
            analyzer = TechnicalAnalyzer(symbol, period='1mo', interval='1d')
            signal = analyzer.generate_signal()
            results[symbol] = {
                'signal': signal.signal,
                'strength': signal.strength,
                'trend': signal.trend
            }
            print(f"✅ {symbol}: {signal.signal} (force {signal.strength}/100)")
            
        except Exception as e:
            print(f"❌ Erreur {symbol}: {e}")
            results[symbol] = {'error': str(e)}
    
    success_count = sum(1 for r in results.values() if 'error' not in r)
    print(f"\n✅ {success_count}/{len(symbols)} symboles analysés avec succès")
    
    return success_count == len(symbols)


def test_existing_imports():
    """Vérifier que les imports existants fonctionnent toujours"""
    print("\n" + "="*70)
    print("🔍 TEST 5: Non-régression (imports existants)")
    print("="*70)
    
    try:
        # Vérifier que les modules existants ne sont pas cassés
        from dashboard.analytics import PortfolioAnalytics
        print("✅ dashboard.analytics importé")
        
        from core.utils import setup_logging
        print("✅ core.utils importé")
        
        from config.tickers import ALL_TICKERS
        print(f"✅ config.tickers importé ({len(ALL_TICKERS)} tickers)")
        
        return True
        
    except Exception as e:
        print(f"❌ Régression détectée: {e}")
        traceback.print_exc()
        return False


def main():
    """Exécuter tous les tests"""
    print("🚀 DÉMARRAGE DES TESTS D'ANALYSE TECHNIQUE")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Analyse basique", test_basic_analysis),
        ("Génération signal", test_signal_generation),
        ("Tous indicateurs", test_all_indicators),
        ("Analyse multiple", test_multiple_symbols),
        ("Non-régression", test_existing_imports)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Erreur fatale dans {test_name}: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Résumé
    print("\n" + "="*70)
    print("📄 RÉSUMÉ DES TESTS")
    print("="*70)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    
    print("\n" + "="*70)
    print(f"🎯 RÉSULTAT FINAL: {passed}/{total} tests réussis")
    print("="*70)
    
    if passed == total:
        print("✅ TOUS LES TESTS PASSÉS - Module prêt pour la production")
        return 0
    else:
        print(f"❌ {total - passed} test(s) échoué(s) - Vérifier les erreurs ci-dessus")
        return 1


if __name__ == '__main__':
    sys.exit(main())
