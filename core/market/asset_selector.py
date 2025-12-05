"""
Sélecteur d'assets universel
Version refactorisée - garde logique existante
"""

from typing import List, Optional
import pandas as pd

from .regime_detector import MarketRegimeDetector
from utils.logger import PloutosLogger

logger = PloutosLogger().get_logger(__name__)

class UniversalAssetSelector:
    """
    Sélectionne les meilleurs assets selon le régime
    """
    
    # Univers fixe par régime
    REGIME_UNIVERSE = {
        'BULL': [
            'NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META',
            'TSLA', 'AMD', 'AVGO', 'NFLX'
        ],
        'BEAR': [
            'SPY', 'QQQ', 'VTI', 'VOO', 'BND', 'TLT',
            'GLD', 'UUP', 'XLP', 'XLU'
        ],
        'SIDEWAYS': [
            'SPY', 'QQQ', 'XLE', 'XLF', 'DIA', 'IWM',
            'XLK', 'XLV', 'XLU', 'XLI'
        ]
    }
    
    def __init__(
        self,
        regime_detector: MarketRegimeDetector,
        enable_market_scan: bool = False
    ):
        """
        Args:
            regime_detector: Détecteur de régime
            enable_market_scan: Activer scan complet (future feature)
        """
        self.regime_detector = regime_detector
        self.enable_market_scan = enable_market_scan
        
        if enable_market_scan:
            logger.warning("Market scan non implémenté, utilisation univers fixe")
    
    def select_assets(
        self,
        n_assets: int = 10,
        lookback_days: int = 90,
        use_market_scan: bool = False
    ) -> List[str]:
        """
        Sélectionne N meilleurs assets
        
        Args:
            n_assets: Nombre d'assets à sélectionner
            lookback_days: Période d'analyse
            use_market_scan: Scan complet marché (non implémenté)
            
        Returns:
            Liste de tickers
        """
        logger.info(f"Sélection de {n_assets} assets")
        
        # Détecter régime si pas déjà fait
        if self.regime_detector.current_regime is None:
            self.regime_detector.detect(lookback_days)
        
        regime = self.regime_detector.current_regime
        logger.info(f"Régime: {regime}")
        
        # Univers selon régime
        universe = self.REGIME_UNIVERSE.get(regime, self.REGIME_UNIVERSE['SIDEWAYS'])
        
        # Limiter à n_assets
        selected = universe[:n_assets]
        
        logger.info(f"Assets sélectionnés: {selected}")
        
        return selected
