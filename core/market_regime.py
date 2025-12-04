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
    
    def __init__(self):
        self.regimes = ['BULL', 'BEAR', 'SIDEWAYS', 'HIGH_VOLATILITY']
        self.current_regime = None
        
    def detect(self):
        """Analyse SPY pour détecter le régime actuel"""
        
        # Télécharger SPY (indice de référence)
        spy = yf.download('SPY', period='90d', interval='1d', progress=False)
        
        # Calculer métriques
        returns = spy['Close'].pct_change().dropna()
        
        # 1. Tendance (MA courte vs MA longue)
        ma_20 = spy['Close'].rolling(20).mean().iloc[-1]
        ma_50 = spy['Close'].rolling(50).mean().iloc[-1]
        trend = (ma_20 - ma_50) / ma_50
        
        # 2. Volatilité (écart-type annualisé)
        volatility = returns.std() * np.sqrt(252)
        
        # 3. Drawdown actuel
        cummax = spy['Close'].cummax()
        drawdown = (spy['Close'].iloc[-1] - cummax.iloc[-1]) / cummax.iloc[-1]
        
        # 4. Momentum (performance 30 derniers jours)
        momentum = (spy['Close'].iloc[-1] - spy['Close'].iloc[-30]) / spy['Close'].iloc[-30]
        
        # LOGIQUE DE DÉTECTION
        if volatility > 0.30:
            regime = 'HIGH_VOLATILITY'
        elif trend > 0.05 and momentum > 0.05:
            regime = 'BULL'
        elif trend < -0.05 or drawdown < -0.10:
            regime = 'BEAR'
        else:
            regime = 'SIDEWAYS'
        
        self.current_regime = regime
        
        return {
            'regime': regime,
            'trend': trend,
            'volatility': volatility,
            'drawdown': drawdown,
            'momentum': momentum,
            'confidence': self._calculate_confidence(trend, volatility, momentum)
        }
    
    def _calculate_confidence(self, trend, volatility, momentum):
        """Score de confiance (0-1) dans la détection"""
        
        # Plus les signaux sont forts et alignés, plus la confiance est haute
        signal_strength = abs(trend) + abs(momentum)
        
        # Pénaliser si volatilité trop haute (incertitude)
        confidence = min(signal_strength / (volatility * 2), 1.0)
        
        return confidence
    
    def get_optimal_strategy(self):
        """Retourne la stratégie optimale selon le régime"""
        
        strategies = {
            'BULL': {
                'asset_selection': 'growth_stocks',  # Tech, momentum
                'position_size': 0.8,                # Agressif
                'stop_loss': 0.15,                   # Large
                'take_profit': 0.30,
                'holding_period': 'long'
            },
            'BEAR': {
                'asset_selection': 'defensive_stocks',  # Utilities, consumer staples
                'position_size': 0.3,                   # Prudent
                'stop_loss': 0.05,                      # Serré
                'take_profit': 0.10,
                'holding_period': 'short'
            },
            'SIDEWAYS': {
                'asset_selection': 'range_bound',    # Mean reversion
                'position_size': 0.5,
                'stop_loss': 0.08,
                'take_profit': 0.15,
                'holding_period': 'medium'
            },
            'HIGH_VOLATILITY': {
                'asset_selection': 'low_beta',       # Minimum variance
                'position_size': 0.2,                # Très prudent
                'stop_loss': 0.03,
                'take_profit': 0.05,
                'holding_period': 'very_short'
            }
        }
        
        return strategies.get(self.current_regime, strategies['SIDEWAYS'])
