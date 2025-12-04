"""
Détecteur de régime de marché automatique
Analyse SPY pour comprendre le contexte macro
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

class MarketRegimeDetector:
    """Détecte automatiquement le régime du marché"""
    
    REGIMES = ['BULL', 'BEAR', 'SIDEWAYS', 'HIGH_VOLATILITY']
    
    def __init__(self, reference_ticker='SPY'):
        self.reference_ticker = reference_ticker
        self.current_regime = None
        self.history = []
        
    def detect(self, lookback_days=90):
        """
        Analyse le marché et retourne le régime actuel
        """
        
        print(f"📊 Analyse du régime de marché ({self.reference_ticker})...")
        
        try:
            # Télécharger données
            spy = yf.download(
                self.reference_ticker, 
                period=f'{lookback_days}d', 
                interval='1d', 
                progress=False
            )
            
            # Si MultiIndex, extraire
            if isinstance(spy.columns, pd.MultiIndex):
                spy = spy.xs(self.reference_ticker, axis=1, level=1)
            
            if len(spy) < 50:
                print("⚠️ Pas assez de données, utilise régime SIDEWAYS par défaut")
                return self._default_regime()
            
            # Calculer métriques (TOUT EN FLOAT)
            returns = spy['Close'].pct_change().dropna()
            
            # 1. Tendance
            ma_20 = float(spy['Close'].rolling(20).mean().iloc[-1])
            ma_50 = float(spy['Close'].rolling(50).mean().iloc[-1])
            trend = (ma_20 - ma_50) / ma_50
            
            # 2. Volatilité
            volatility = float(returns.std() * np.sqrt(252))
            
            # 3. Drawdown
            cummax = spy['Close'].cummax()
            current_drawdown = float((spy['Close'].iloc[-1] - cummax.iloc[-1]) / cummax.iloc[-1])
            
            # 4. Momentum
            if len(spy) >= 30:
                momentum = float((spy['Close'].iloc[-1] - spy['Close'].iloc[-30]) / spy['Close'].iloc[-30])
            else:
                momentum = 0.0
            
            # 5. Volume
            volume_ma_20 = float(spy['Volume'].rolling(20).mean().iloc[-1])
            volume_current = float(spy['Volume'].iloc[-5:].mean())
            volume_trend = (volume_current - volume_ma_20) / volume_ma_20
            
            # Classification
            regime = self._classify_regime(trend, volatility, current_drawdown, momentum, volume_trend)
            confidence = self._calculate_confidence(trend, volatility, momentum)
            
            result = {
                'regime': regime,
                'trend': float(trend),
                'volatility': float(volatility),
                'drawdown': float(current_drawdown),
                'momentum': float(momentum),
                'volume_trend': float(volume_trend),
                'confidence': float(confidence),
                'timestamp': datetime.now().isoformat()
            }
            
            self.current_regime = regime
            self.history.append(result)
            
            # Affichage
            print(f"  🎯 Régime détecté : {regime} (confiance: {confidence:.1%})")
            print(f"  📈 Trend: {trend:+.2%} | Vol: {volatility:.1%} | DD: {current_drawdown:.1%}")
            
            return result
            
        except Exception as e:
            print(f"⚠️ Erreur détection régime : {e}")
            return self._default_regime()
    
    def _classify_regime(self, trend, volatility, drawdown, momentum, volume_trend):
        """Classifie le régime selon les métriques (tous des floats)"""
        
        # Haute volatilité = priorité
        if volatility > 0.30:
            return 'HIGH_VOLATILITY'
        
        # Bull market
        if trend > 0.05 and momentum > 0.05:
            return 'BULL'
        
        # Bear market
        if trend < -0.05 or drawdown < -0.10:
            return 'BEAR'
        
        # Sideways
        return 'SIDEWAYS'
    
    def _calculate_confidence(self, trend, volatility, momentum):
        """Score de confiance (0-1)"""
        
        # Force des signaux
        signal_strength = abs(trend) + abs(momentum)
        
        # Cohérence
        coherence = 1.0 if (trend * momentum) > 0 else 0.5
        
        # Pénalité volatilité
        volatility_penalty = max(0, 1 - (volatility / 0.5))
        
        confidence = min(signal_strength * coherence * volatility_penalty, 1.0)
        
        return float(confidence)
    
    def _default_regime(self):
        """Régime par défaut"""
        return {
            'regime': 'SIDEWAYS',
            'trend': 0.0,
            'volatility': 0.20,
            'drawdown': 0.0,
            'momentum': 0.0,
            'volume_trend': 0.0,
            'confidence': 0.5,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_optimal_strategy(self, regime=None):
        """Retourne la stratégie optimale selon le régime"""
        
        if regime is None:
            regime = self.current_regime if self.current_regime else 'SIDEWAYS'
        
        strategies = {
            'BULL': {
                'asset_selection': 'growth_stocks',
                'position_size': 0.80,
                'stop_loss': 0.15,
                'take_profit': 0.30,
                'holding_period': 'long',
                'max_positions': 5,
                'rebalance_frequency': 'weekly'
            },
            'BEAR': {
                'asset_selection': 'defensive_stocks',
                'position_size': 0.30,
                'stop_loss': 0.05,
                'take_profit': 0.10,
                'holding_period': 'short',
                'max_positions': 3,
                'rebalance_frequency': 'daily'
            },
            'SIDEWAYS': {
                'asset_selection': 'range_bound',
                'position_size': 0.50,
                'stop_loss': 0.08,
                'take_profit': 0.15,
                'holding_period': 'medium',
                'max_positions': 4,
                'rebalance_frequency': 'weekly'
            },
            'HIGH_VOLATILITY': {
                'asset_selection': 'low_beta',
                'position_size': 0.20,
                'stop_loss': 0.03,
                'take_profit': 0.05,
                'holding_period': 'very_short',
                'max_positions': 2,
                'rebalance_frequency': 'daily'
            }
        }
        
        return strategies.get(regime, strategies['SIDEWAYS'])
    
    def save_history(self, filepath='data/regime_history.csv'):
        """Sauvegarde l'historique"""
        if len(self.history) > 0:
            df = pd.DataFrame(self.history)
            df.to_csv(filepath, index=False)
            print(f"💾 Historique sauvegardé : {filepath}")
