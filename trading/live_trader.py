"""Live Trader - Version refactorisée"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading.broker_factory import create_broker
from trading.brain_trader import BrainTrader
from trading.portfolio_manager import PortfolioManager
from trading.stop_loss_manager import StopLossManager
from core.utils import setup_logging
from datetime import datetime, date
import os
import time

logger = setup_logging(__name__, 'live_trader.log')

# Imports optionnels
try:
    from database.db import log_prediction, save_daily_summary
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

try:
    from core.risk_manager import RiskManager
    RISK_AVAILABLE = True
except ImportError:
    RISK_AVAILABLE = False

try:
    from core.monitoring import start_monitoring
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

class LiveTrader:
    """Trader live simplifié"""
    
    def __init__(self, paper_trading=True, capital=None, monitoring_port=9090, broker=None):
        self.paper_trading = paper_trading

        # Monitoring
        self.metrics = None
        if MONITORING_AVAILABLE:
            self.metrics = start_monitoring(port=monitoring_port)
            logger.info(f"✅ Monitoring: http://localhost:{monitoring_port}/metrics")

        # Client broker (eToro par défaut, ou Alpaca)
        broker_name = broker or os.getenv('BROKER', 'etoro')
        self.broker = create_broker(broker_name, paper_trading=paper_trading)
        # Alias pour compatibilité
        self.alpaca = self.broker
        self.brain = BrainTrader(capital=capital, paper_trading=paper_trading)
        
        # Gestionnaires
        account = self.broker.get_account()
        self.initial_capital = account['portfolio_value']

        self.portfolio_mgr = PortfolioManager(self.broker, self.initial_capital)
        self.stop_loss_mgr = StopLossManager(self.broker)
        
        if RISK_AVAILABLE:
            self.risk_manager = RiskManager()
            self.risk_manager.reset_daily_stats(self.initial_capital)
        else:
            self.risk_manager = None
        
        logger.info(f"💰 Portfolio: ${self.initial_capital:,.2f}")
        logger.info(f"🛡️ Risk Management: {'✅' if RISK_AVAILABLE else '❌'}")
    
    def check_risk_management(self):
        """Vérifier risques et stop loss"""
        positions = self.alpaca.get_positions()
        
        # Stop Loss / Take Profit
        self.stop_loss_mgr.check_all_positions(positions, self.metrics)
        
        # Circuit breaker
        if RISK_AVAILABLE:
            account = self.alpaca.get_account()
            if not self.risk_manager.check_daily_loss_limit(account['portfolio_value']):
                logger.error("🚨 CIRCUIT BREAKER ACTIF")
                for pos in positions:
                    self.alpaca.close_position(pos['symbol'], reason='Circuit Breaker')
    
    def execute_signals(self):
        """Exécuter les signaux de trading"""
        logger.info(f"\n🧠 ANALYSE - {datetime.now().strftime('%H:%M:%S')}")
        
        predictions = self.brain.predict_all()
        account = self.alpaca.get_account()
        portfolio_value = account['portfolio_value']
        
        current_positions_dict = {
            pos['symbol']: pos 
            for pos in self.alpaca.get_positions()
        }
        
        for sector, sector_preds in predictions.items():
            for pred in sector_preds:
                symbol = pred['ticker']
                action = pred['action']
                
                current_price = self.alpaca.get_current_price(symbol)
                if not current_price:
                    continue
                
                position = current_positions_dict.get(symbol)
                
                # BUY signal
                if action == 'BUY':
                    if position:
                        # Renforcer
                        should_add, reason = self.portfolio_mgr.should_add_to_position(
                            symbol, position, portfolio_value
                        )
                        
                        if should_add:
                            qty = self.portfolio_mgr.calculate_position_size(
                                symbol, current_price, portfolio_value, self.risk_manager
                            )
                            
                            if qty > 0:
                                self.alpaca.place_market_order(symbol, qty, 'buy', reason='Renforcement')
                    else:
                        # Nouvelle position
                        qty = self.portfolio_mgr.calculate_position_size(
                            symbol, current_price, portfolio_value, self.risk_manager
                        )
                        
                        if qty > 0:
                            self.alpaca.place_market_order(symbol, qty, 'buy', reason='Nouvelle position')
                
                # SELL signal
                elif action == 'SELL' and position:
                    self.alpaca.close_position(symbol, reason='Signal SELL AI')
        
        logger.info("✅ Cycle terminé")
    
    def run(self, check_interval_minutes=60):
        """Boucle principale"""
        logger.info("🚀 LIVE TRADER DÉMARRÉ")
        
        try:
            while True:
                self.check_risk_management()
                self.execute_signals()
                
                logger.info(f"⏳ Prochain cycle dans {check_interval_minutes} min...")
                time.sleep(check_interval_minutes * 60)
        
        except KeyboardInterrupt:
            logger.info("🛑 Arrêt manuel")
        except Exception as e:
            logger.error(f"❌ Erreur: {e}", exc_info=True)

if __name__ == "__main__":
    trader = LiveTrader(paper_trading=True)
    trader.run(check_interval_minutes=60)
