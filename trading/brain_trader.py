# trading/brain_trader.py
"""Brain Trader utilisant l'architecture modulaire"""

# === FIX PATH ===
import sys
from pathlib import Path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))
# ================

from config.settings import TRADING_CONFIG
from config.tickers import SECTORS, TICKER_TO_SECTOR
from core.features import FeatureCalculator
from core.models import ModelManager
from core.utils import setup_logging
from datetime import datetime
import random

logger = setup_logging(__name__, 'trading.log')

class BrainTrader:
    """Trader multi-secteurs"""
    
    def __init__(self, capital=None, paper_trading=True):
        self.capital = capital or TRADING_CONFIG['initial_capital']
        self.paper_trading = paper_trading
        
        # Charger modèles
        self.model_manager = ModelManager()
        self.models = {}
        self.demo_mode = True
        
        for sector, config in SECTORS.items():
            model_name = config['model_name']
            model = self.model_manager.load_model(model_name)
            if model:
                self.models[sector] = model
                self.demo_mode = False
                logger.info(f"✅ {sector}: {model_name}")
            else:
                logger.warning(f"⚠️  {sector}: Modèle non chargé")
        
        if self.demo_mode:
            logger.warning("⚠️  MODE DÉMO - Décisions aléatoires")
        
        logger.info(f"💰 Capital: ${self.capital:,}")
        logger.info(f"📊 Mode: {'Paper' if paper_trading else 'Live'}")
    
    def predict(self, ticker: str):
        """Prédire l'action pour un ticker"""
        sector = TICKER_TO_SECTOR.get(ticker)
        if not sector:
            logger.warning(f"⚠️  {ticker}: Secteur inconnu")
            return None
        
        # Mode démo ou pas de modèle
        if self.demo_mode or sector not in self.models:
            action_int = random.randint(0, 2)
        else:
            # Mode réel avec modèle
            model = self.models[sector]
            n_features = model.observation_space.shape[0]
            features = FeatureCalculator.calculate(ticker, n_features)
            
            if features is None:
                return None
            
            action, _ = model.predict(features, deterministic=True)
            
            try:
                action_int = int(action)
            except:
                try:
                    action_int = int(action.item())
                except:
                    action_int = 0
        
        action_name = ['HOLD', 'BUY', 'SELL'][action_int]
        
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
