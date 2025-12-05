"""
Détecteur de régime de marché
Version refactorisée - garde logique existante
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict

from utils.logger import PloutosLogger

logger = PloutosLogger().get_logger(__name__)

class MarketRegimeDetector:
    """
    Détecte le régime de marché (BULL, BEAR, SIDEWAYS)
    """
    
    def __init__(self, reference_ticker: str = 'SPY'):
        """
        Args:
            reference_ticker: Ticker de référence (SPY par défaut)
        """
        self.reference_ticker = reference_ticker
        self.current_regime = None
        self.confidence = 0.0
    
    def detect(self, lookback_days: int = 90) -> Dict[str, any]:
        """
        Détecte le régime actuel
        
        Args:
            lookback_days: Période d'analyse
            
        Returns:
            Dict avec 'regime' et 'confidence'
        """
        logger.info(f"Détection régime sur {lookback_days} jours ({self.reference_ticker})")
        
        try:
            # Télécharger données
            data = yf.download(
                self.reference_ticker,
                period=f'{lookback_days}d',
                interval='1d',
                progress=False
            )
            
            if data.empty or len(data) < 20:
                logger.warning("Données insuffisantes, régime par défaut SIDEWAYS")
                return {'regime': 'SIDEWAYS', 'confidence': 0.5}
            
            # Calculer indicateurs
            closes = data['Close']
            returns = closes.pct_change().dropna()
            
            # Tendance (MA court vs MA long)
            ma_short = closes.rolling(20).mean().iloc[-1]
            ma_long = closes.rolling(50).mean().iloc[-1]
            trend = (ma_short / ma_long) - 1
            
            # Volatilité
            volatility = returns.std()
            
            # Momentum
            momentum = (closes.iloc[-1] / closes.iloc[0]) - 1
            
            # Détection régime
            if momentum > 0.15 and trend > 0.05:
                regime = 'BULL'
                confidence = min(abs(momentum) + abs(trend), 1.0)
                
            elif momentum < -0.15 and trend < -0.05:
                regime = 'BEAR'
                confidence = min(abs(momentum) + abs(trend), 1.0)
                
            else:
                regime = 'SIDEWAYS'
                confidence = 1 - (abs(momentum) + abs(trend))
            
            self.current_regime = regime
            self.confidence = confidence
            
            logger.info(f"Régime détecté: {regime} (confiance: {confidence:.1%})")
            
            return {
                'regime': regime,
                'confidence': confidence,
                'momentum': momentum,
                'trend': trend,
                'volatility': volatility
            }
            
        except Exception as e:
            logger.error(f"Erreur détection régime: {e}")
            return {'regime': 'SIDEWAYS', 'confidence': 0.5}
