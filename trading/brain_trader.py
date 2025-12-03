# trading/brain_trader.py
"""Brain Trader utilisant l'architecture modulaire"""

# FIX: Ajouter le projet au path
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import TRADING_CONFIG
from config.tickers import SECTORS, TICKER_TO_SECTOR
from core.features import FeatureCalculator
from core.models import ModelManager
from core.utils import setup_logging

from datetime import datetime

logger = setup_logging(__name__, 'trading.log')

class BrainTrader:
    """Trader multi-secteurs"""
    
    def __init__(self, capital=None, paper_trading=True):
        self.capital = capital or TRADING_CONFIG['initial_capital']
        self.paper_trading = paper_trading
        
        # Charger modèles
        self.model_manager = ModelManager()
        self.models = {}
        
        for sector, config in SECTORS.items():
            model_name = config['model_name']
            model = self.model_manager.load_model(model_name)
            if model:
                self.models[sector] = model
                logger.info(f"✅ {sector}: {model_name}")
            else:
                logger.warning(f"⚠️  {sector}: Modèle non chargé")
        
        logger.info(f"💰 Capital: ${self.capital:,}")
        logger.info(f"📊 Mode: {'Paper' if paper_trading else 'Live'}")
    
    def predict(self, ticker: str):
        """Prédire l'action pour un ticker"""
        # Déterminer le secteur
        sector = TICKER_TO_SECTOR.get(ticker)
        if not sector or sector not in self.models:
            logger.warning(f"⚠️  {ticker}: Pas de modèle pour ce secteur")
            return None
        
        # Calculer features
        model = self.models[sector]
        n_features = model.observation_space.shape[0]
        features = FeatureCalculator.calculate(ticker, n_features)
        
        if features is None:
            return None
        
        # Prédiction
        action, _ = model.predict(features, deterministic=True)
        
        # Conversion safe
        try:
            action_int = int(action)
        except:
            try:
                action_int = int(action.item())
            except:
                action_int = 0
        
        action_name = ['HOLD', 'BUY', 'SELL'][action_int]
        
        # Allocation
        sector_config = SECTORS[sector]
        allocation = sector_config['allocation'] / len(sector_config['tickers'])
        capital_allocated = self.capital * allocation
        
        return {
            'ticker': ticker,
            'action': action_name,
            'sector': sector,
            'capital': capital_allocated,
            'timestamp': datetime.now()
        }
    
    def predict_all(self):
        """Prédire pour tous les secteurs"""
        results = {}
        
        for sector, config in SECTORS.items():
            results[sector] = []
            
            for ticker in config['tickers']:
                pred = self.predict(ticker)
                if pred:
                    results[sector].append(pred)
        
        return results
