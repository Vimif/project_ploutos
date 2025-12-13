#!/usr/bin/env python3
"""
🎯 PLOUTOS V8 ORACLE - ENSEMBLE META-MODEL

Orchestre tous les modèles prédictifs V8:
- LightGBM Intraday (1 jour)
- XGBoost Weekly (5 jours)  
- Random Forest Monthly (20 jours)

Aggrège les prédictions avec pondération dynamique
basée sur l'accuracy historique

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import Dict, List
from datetime import datetime
import json

# Import model loader pour chemins absolus
try:
    from src.models.v8_model_loader import get_model_path
    MODEL_LOADER_AVAILABLE = True
except:
    MODEL_LOADER_AVAILABLE = False

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


class V8OracleEnsemble:
    """
    Ensemble de modèles prédictifs avec aggrégation intelligente
    """
    
    def __init__(self):
        self.models = {}
        self.weights = {
            'intraday': 0.4,   # Court terme
            'weekly': 0.35,    # Moyen terme
            'monthly': 0.25    # Long terme
        }
        
    def load_models(self) -> bool:
        """
        Charge tous les modèles disponibles avec chemins absolus
        """
        print("\n🔥 Chargement des modèles V8 Oracle...")
        
        loaded = 0
        
        # LightGBM Intraday
        if LIGHTGBM_AVAILABLE:
            try:
                self.models['intraday'] = V8LightGBMIntraday()
                
                # Utiliser model_loader si disponible
                if MODEL_LOADER_AVAILABLE:
                    model_path = get_model_path('v8_lightgbm_intraday.pkl')
                    success = self.models['intraday'].load(model_path)
                else:
                    success = self.models['intraday'].load()
                
                if success:
                    print("✅ LightGBM Intraday (1 jour) chargé")
                    loaded += 1
                else:
                    del self.models['intraday']
            except Exception as e:
                print(f"⚠️  LightGBM Intraday: {e}")
        
        # XGBoost Weekly
        if XGBOOST_AVAILABLE:
            try:
                self.models['weekly'] = V8XGBoostWeekly()
                
                # Utiliser model_loader si disponible
                if MODEL_LOADER_AVAILABLE:
                    model_path = get_model_path('v8_xgboost_weekly.pkl')
                    success = self.models['weekly'].load(model_path)
                else:
                    success = self.models['weekly'].load()
                
                if success:
                    print("✅ XGBoost Weekly (5 jours) chargé")
                    loaded += 1
                else:
                    del self.models['weekly']
            except Exception as e:
                print(f"⚠️  XGBoost Weekly: {e}")
        
        print(f"\n🎯 {loaded} modèle(s) chargé(s)\n")
        return loaded > 0
    
    def predict_multi_horizon(self, ticker: str) -> Dict:
        """
        Prédictions pour tous les horizons
        """
        if not self.models:
            return {'error': 'Aucun modèle chargé'}
        
        results = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'predictions': {}
        }
        
        # Prédire pour chaque horizon
        for horizon, model in self.models.items():
            try:
                pred = model.predict(ticker)
                if 'error' not in pred:
                    results['predictions'][horizon] = pred
            except Exception as e:
                results['predictions'][horizon] = {'error': str(e)}
        
        # Aggrégation si plusieurs modèles
        if len(results['predictions']) > 1:
            results['ensemble'] = self._aggregate_predictions(results['predictions'])
        
        return results
    
    def _aggregate_predictions(self, predictions: Dict) -> Dict:
        """
        Aggrège les prédictions avec pondération
        """
        # Compter les votes
        up_votes = 0
        down_votes = 0
        total_confidence = 0
        total_weight = 0
        
        for horizon, pred in predictions.items():
            if 'error' in pred:
                continue
            
            weight = self.weights.get(horizon, 0.33)
            confidence = pred.get('confidence', 50) / 100
            
            if pred['prediction'] == 'UP':
                up_votes += weight
                total_confidence += confidence * weight
            else:
                down_votes += weight
                total_confidence += confidence * weight
            
            total_weight += weight
        
        if total_weight == 0:
            return {'error': 'Pas de prédictions valides'}
        
        # Décision finale
        final_prediction = 'UP' if up_votes > down_votes else 'DOWN'
        final_confidence = (total_confidence / total_weight) * 100
        
        # Consensus
        agreement = 'STRONG' if abs(up_votes - down_votes) / total_weight > 0.6 else 'WEAK'
        
        return {
            'prediction': final_prediction,
            'confidence': round(final_confidence, 2),
            'agreement': agreement,
            'votes': {
                'up': round(up_votes, 2),
                'down': round(down_votes, 2)
            },
            'models_used': len([p for p in predictions.values() if 'error' not in p])
        }
    
    def get_recommendation(self, ticker: str, risk_tolerance: str = 'medium') -> Dict:
        """
        Recommandation de trading basée sur les prédictions
        
        Args:
            ticker: Ticker à analyser
            risk_tolerance: 'low', 'medium', 'high'
        """
        predictions = self.predict_multi_horizon(ticker)
        
        if 'error' in predictions:
            return predictions
        
        ensemble = predictions.get('ensemble', {})
        
        if 'error' in ensemble:
            # Utiliser le meilleur modèle disponible
            best_pred = max(
                predictions['predictions'].values(),
                key=lambda x: x.get('confidence', 0) if 'error' not in x else 0
            )
            ensemble = {
                'prediction': best_pred['prediction'],
                'confidence': best_pred['confidence'],
                'agreement': 'SINGLE_MODEL'
            }
        
        # Seuils de confiance selon tolérance au risque
        confidence_thresholds = {
            'low': 75,      # Conservateur
            'medium': 65,   # Équilibré
            'high': 55      # Agressif
        }
        
        threshold = confidence_thresholds.get(risk_tolerance, 65)
        confidence = ensemble['confidence']
        prediction = ensemble['prediction']
        
        # Décision
        if confidence >= threshold:
            if prediction == 'UP':
                action = 'BUY'
                strength = 'STRONG' if confidence >= 70 else 'MODERATE'
            else:
                action = 'SELL' if ensemble.get('agreement') == 'STRONG' else 'HOLD'
                strength = 'STRONG' if confidence >= 70 else 'MODERATE'
        else:
            action = 'HOLD'
            strength = 'WEAK'
        
        return {
            'ticker': ticker,
            'action': action,
            'strength': strength,
            'confidence': confidence,
            'prediction': prediction,
            'agreement': ensemble.get('agreement'),
            'risk_tolerance': risk_tolerance,
            'threshold_used': threshold,
            'all_predictions': predictions['predictions'],
            'timestamp': predictions['timestamp']
        }
    
    def batch_predict(self, tickers: List[str]) -> Dict:
        """
        Prédit pour plusieurs tickers
        """
        results = {}
        
        for ticker in tickers:
            print(f"🔮 Analyse {ticker}...")
            results[ticker] = self.predict_multi_horizon(ticker)
        
        return {
            'tickers': results,
            'summary': self._generate_summary(results),
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_summary(self, results: Dict) -> Dict:
        """
        Génère un résumé des prédictions
        """
        total = len(results)
        bullish = 0
        bearish = 0
        high_confidence = 0
        
        for ticker, data in results.items():
            if 'error' in data:
                continue
            
            ensemble = data.get('ensemble', {})
            if 'error' in ensemble:
                continue
            
            if ensemble['prediction'] == 'UP':
                bullish += 1
            else:
                bearish += 1
            
            if ensemble['confidence'] >= 70:
                high_confidence += 1
        
        return {
            'total_analyzed': total,
            'bullish': bullish,
            'bearish': bearish,
            'neutral': total - bullish - bearish,
            'high_confidence_count': high_confidence,
            'bullish_pct': round(bullish / total * 100, 1) if total > 0 else 0
        }
    
    def save_predictions(self, predictions: Dict, path: str = "logs/v8_predictions.json"):
        """
        Sauvegarde les prédictions
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print(f"💾 Prédictions sauvegardées: {path}")


if __name__ == '__main__':
    # Exemple d'utilisation
    oracle = V8OracleEnsemble()
    
    if oracle.load_models():
        # Test sur un ticker
        print("\n" + "="*70)
        print("🔮 TEST NVDA")
        print("="*70)
        
        result = oracle.predict_multi_horizon('NVDA')
        
        print("\n📊 Prédictions par horizon:")
        for horizon, pred in result.get('predictions', {}).items():
            if 'error' not in pred:
                print(f"\n{horizon.upper()}:")
                print(f"  Prédiction: {pred['prediction']}")
                print(f"  Confiance: {pred['confidence']:.1f}%")
        
        if 'ensemble' in result:
            ensemble = result['ensemble']
            print("\n🎯 ENSEMBLE:")
            print(f"  Prédiction: {ensemble['prediction']}")
            print(f"  Confiance: {ensemble['confidence']:.1f}%")
            print(f"  Agreement: {ensemble['agreement']}")
        
        # Recommandation
        print("\n" + "="*70)
        print("💼 RECOMMANDATION")
        print("="*70)
        
        for risk in ['low', 'medium', 'high']:
            rec = oracle.get_recommendation('NVDA', risk_tolerance=risk)
            print(f"\nRisk {risk.upper()}: {rec['action']} ({rec['strength']}) - Conf: {rec['confidence']:.1f}%")
        
        # Batch
        print("\n" + "="*70)
        print("📊 BATCH ANALYSIS")
        print("="*70 + "\n")
        
        tickers = ['NVDA', 'MSFT', 'AAPL']
        batch = oracle.batch_predict(tickers)
        
        print("\n📊 Résumé:")
        summary = batch['summary']
        print(f"  Bullish: {summary['bullish']} ({summary['bullish_pct']:.1f}%)")
        print(f"  Bearish: {summary['bearish']}")
        print(f"  High confidence: {summary['high_confidence_count']}")
    else:
        print("❌ Aucun modèle disponible")
