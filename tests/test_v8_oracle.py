#!/usr/bin/env python3
"""
🧪 TESTS V8 ORACLE SYSTEM

Tests complets pour le système V8 Oracle

Usage:
    python tests/test_v8_oracle.py
    python tests/test_v8_oracle.py --quick
    python tests/test_v8_oracle.py --model lightgbm

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
from datetime import datetime

try:
    from src.models.v8_lightgbm_intraday import V8LightGBMIntraday
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False

try:
    from src.models.v8_xgboost_weekly import V8XGBoostWeekly
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

try:
    from src.models.v8_oracle_ensemble import V8OracleEnsemble
    ENSEMBLE_AVAILABLE = True
except:
    ENSEMBLE_AVAILABLE = False


class V8OracleTester:
    """
    Testeur complet pour V8 Oracle
    """
    
    def __init__(self):
        self.results = []
    
    def test_lightgbm(self) -> bool:
        """
        Test LightGBM Intraday
        """
        print("\n" + "="*70)
        print("🛠️  TEST 1: LIGHTGBM INTRADAY (1 JOUR)")
        print("="*70)
        
        if not LIGHTGBM_AVAILABLE:
            print("❌ LightGBM non disponible")
            return False
        
        try:
            predictor = V8LightGBMIntraday()
            
            # Charger modèle
            if predictor.load():
                print("✅ Modèle chargé")
            else:
                print("⚠️  Modèle non trouvé, entraînement requis")
                return False
            
            # Test prédiction
            print("\n🔮 Test prédiction NVDA...")
            start = time.time()
            result = predictor.predict('NVDA')
            elapsed = (time.time() - start) * 1000
            
            if 'error' not in result:
                print(f"\n✅ Prédiction réussie ({elapsed:.1f}ms)")
                print(f"   Ticker: {result['ticker']}")
                print(f"   Prédiction: {result['prediction']}")
                print(f"   Confiance: {result['confidence']:.2f}%")
                print(f"   Prix actuel: ${result['current_price']:.2f}")
                
                self.results.append({
                    'test': 'LightGBM Intraday',
                    'status': 'PASS',
                    'latency_ms': elapsed
                })
                return True
            else:
                print(f"\n❌ Erreur: {result['error']}")
                self.results.append({
                    'test': 'LightGBM Intraday',
                    'status': 'FAIL',
                    'error': result['error']
                })
                return False
                
        except Exception as e:
            print(f"\n❌ Exception: {e}")
            self.results.append({
                'test': 'LightGBM Intraday',
                'status': 'ERROR',
                'error': str(e)
            })
            return False
    
    def test_xgboost(self) -> bool:
        """
        Test XGBoost Weekly
        """
        print("\n" + "="*70)
        print("🛠️  TEST 2: XGBOOST WEEKLY (5 JOURS)")
        print("="*70)
        
        if not XGBOOST_AVAILABLE:
            print("❌ XGBoost non disponible")
            return False
        
        try:
            predictor = V8XGBoostWeekly()
            
            if predictor.load():
                print("✅ Modèle chargé")
            else:
                print("⚠️  Modèle non trouvé, entraînement requis")
                return False
            
            print("\n🔮 Test prédiction MSFT...")
            start = time.time()
            result = predictor.predict('MSFT')
            elapsed = (time.time() - start) * 1000
            
            if 'error' not in result:
                print(f"\n✅ Prédiction réussie ({elapsed:.1f}ms)")
                print(f"   Ticker: {result['ticker']}")
                print(f"   Horizon: {result['horizon']}")
                print(f"   Prédiction: {result['prediction']}")
                print(f"   Confiance: {result['confidence']:.2f}%")
                
                self.results.append({
                    'test': 'XGBoost Weekly',
                    'status': 'PASS',
                    'latency_ms': elapsed
                })
                return True
            else:
                print(f"\n❌ Erreur: {result['error']}")
                self.results.append({
                    'test': 'XGBoost Weekly',
                    'status': 'FAIL',
                    'error': result['error']
                })
                return False
                
        except Exception as e:
            print(f"\n❌ Exception: {e}")
            self.results.append({
                'test': 'XGBoost Weekly',
                'status': 'ERROR',
                'error': str(e)
            })
            return False
    
    def test_ensemble(self) -> bool:
        """
        Test Ensemble Oracle
        """
        print("\n" + "="*70)
        print("🛠️  TEST 3: ENSEMBLE ORACLE (MULTI-HORIZON)")
        print("="*70)
        
        if not ENSEMBLE_AVAILABLE:
            print("❌ Ensemble non disponible")
            return False
        
        try:
            oracle = V8OracleEnsemble()
            
            if oracle.load_models():
                print(f"✅ {len(oracle.models)} modèle(s) chargé(s)")
            else:
                print("⚠️  Aucun modèle chargé")
                return False
            
            # Test prédiction multi-horizon
            print("\n🔮 Test prédiction multi-horizon AAPL...")
            start = time.time()
            result = oracle.predict_multi_horizon('AAPL')
            elapsed = (time.time() - start) * 1000
            
            if 'error' not in result:
                print(f"\n✅ Prédiction réussie ({elapsed:.1f}ms)")
                
                for horizon, pred in result.get('predictions', {}).items():
                    if 'error' not in pred:
                        print(f"\n   {horizon.upper()}:")
                        print(f"      Prédiction: {pred['prediction']}")
                        print(f"      Confiance: {pred['confidence']:.2f}%")
                
                if 'ensemble' in result:
                    ens = result['ensemble']
                    print(f"\n   ENSEMBLE:")
                    print(f"      Prédiction: {ens['prediction']}")
                    print(f"      Confiance: {ens['confidence']:.2f}%")
                    print(f"      Agreement: {ens['agreement']}")
                
                self.results.append({
                    'test': 'Ensemble Oracle',
                    'status': 'PASS',
                    'latency_ms': elapsed,
                    'models_used': len(result.get('predictions', {}))
                })
                return True
            else:
                print(f"\n❌ Erreur: {result['error']}")
                self.results.append({
                    'test': 'Ensemble Oracle',
                    'status': 'FAIL',
                    'error': result['error']
                })
                return False
                
        except Exception as e:
            print(f"\n❌ Exception: {e}")
            self.results.append({
                'test': 'Ensemble Oracle',
                'status': 'ERROR',
                'error': str(e)
            })
            return False
    
    def test_recommendations(self) -> bool:
        """
        Test Recommandations
        """
        print("\n" + "="*70)
        print("🛠️  TEST 4: RECOMMANDATIONS DE TRADING")
        print("="*70)
        
        if not ENSEMBLE_AVAILABLE:
            print("❌ Ensemble non disponible")
            return False
        
        try:
            oracle = V8OracleEnsemble()
            oracle.load_models()
            
            print("\n🔮 Test recommandations GOOGL...")
            
            for risk in ['low', 'medium', 'high']:
                rec = oracle.get_recommendation('GOOGL', risk_tolerance=risk)
                
                if 'error' not in rec:
                    print(f"\n   Risk {risk.upper()}:")
                    print(f"      Action: {rec['action']}")
                    print(f"      Strength: {rec['strength']}")
                    print(f"      Confiance: {rec['confidence']:.2f}%")
            
            self.results.append({
                'test': 'Recommendations',
                'status': 'PASS'
            })
            return True
            
        except Exception as e:
            print(f"\n❌ Exception: {e}")
            self.results.append({
                'test': 'Recommendations',
                'status': 'ERROR',
                'error': str(e)
            })
            return False
    
    def test_batch(self) -> bool:
        """
        Test Batch Predictions
        """
        print("\n" + "="*70)
        print("🛠️  TEST 5: BATCH PREDICTIONS")
        print("="*70)
        
        if not ENSEMBLE_AVAILABLE:
            print("❌ Ensemble non disponible")
            return False
        
        try:
            oracle = V8OracleEnsemble()
            oracle.load_models()
            
            tickers = ['NVDA', 'MSFT', 'AAPL']
            
            print(f"\n🔮 Test batch {len(tickers)} tickers...")
            start = time.time()
            result = oracle.batch_predict(tickers)
            elapsed = (time.time() - start) * 1000
            
            print(f"\n✅ Batch réussi ({elapsed:.1f}ms)")
            
            summary = result['summary']
            print(f"\n   Résumé:")
            print(f"      Total: {summary['total_analyzed']}")
            print(f"      Bullish: {summary['bullish']} ({summary['bullish_pct']:.1f}%)")
            print(f"      Bearish: {summary['bearish']}")
            print(f"      High confidence: {summary['high_confidence_count']}")
            
            self.results.append({
                'test': 'Batch Predictions',
                'status': 'PASS',
                'latency_ms': elapsed,
                'tickers': len(tickers)
            })
            return True
            
        except Exception as e:
            print(f"\n❌ Exception: {e}")
            self.results.append({
                'test': 'Batch Predictions',
                'status': 'ERROR',
                'error': str(e)
            })
            return False
    
    def print_summary(self):
        """
        Affiche le résumé des tests
        """
        print("\n" + "="*70)
        print("📊 RÉSUMÉ DES TESTS")
        print("="*70)
        
        total = len(self.results)
        passed = len([r for r in self.results if r['status'] == 'PASS'])
        failed = len([r for r in self.results if r['status'] in ['FAIL', 'ERROR']])
        
        print(f"\nTotal: {total}")
        print(f"Passés: {passed} ✅")
        print(f"Échoués: {failed} {'❌' if failed > 0 else ''}")
        
        print("\n📊 Détail:")
        for result in self.results:
            status_icon = '✅' if result['status'] == 'PASS' else '❌'
            latency = f" ({result.get('latency_ms', 0):.1f}ms)" if 'latency_ms' in result else ''
            print(f"   {status_icon} {result['test']}{latency}")
        
        print("\n" + "="*70 + "\n")
        
        return passed == total


def main():
    parser = argparse.ArgumentParser(description='Tests V8 Oracle')
    parser.add_argument('--quick', action='store_true',
                       help='Tests rapides uniquement')
    parser.add_argument('--model', type=str, default='all',
                       help='Modèle à tester (lightgbm, xgboost, ensemble, all)')
    
    args = parser.parse_args()
    
    tester = V8OracleTester()
    
    print("\n" + "="*70)
    print("🧪 TESTS V8 ORACLE SYSTEM")
    print("="*70)
    print(f"\nMode: {'QUICK' if args.quick else 'COMPLET'}")
    print(f"Modèles: {args.model}\n")
    
    # Exécuter tests
    if args.model in ['all', 'lightgbm']:
        tester.test_lightgbm()
    
    if args.model in ['all', 'xgboost']:
        tester.test_xgboost()
    
    if args.model in ['all', 'ensemble']:
        tester.test_ensemble()
        
        if not args.quick:
            tester.test_recommendations()
            tester.test_batch()
    
    # Résumé
    success = tester.print_summary()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
