# trading/live_trader.py
"""Trader en temps réel avec Alpaca"""

# === FIX PATH ===
import sys
from pathlib import Path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))
# ================

from trading.alpaca_client import AlpacaClient
from trading.brain_trader import BrainTrader
from config.settings import TRADING_CONFIG
from config.tickers import SECTORS, ALL_TICKERS
from core.utils import setup_logging
from datetime import datetime
import time

logger = setup_logging(__name__, 'live_trader.log')

class LiveTrader:
    """Trader live avec Alpaca + Brain AI"""
    
    def __init__(self, paper_trading=True, capital=None):
        """
        Initialiser le trader live
        
        Args:
            paper_trading: True pour paper, False pour live
            capital: Capital initial (optionnel)
        """
        self.paper_trading = paper_trading
        
        # Clients
        self.alpaca = AlpacaClient(paper_trading=paper_trading)
        self.brain = BrainTrader(capital=capital, paper_trading=paper_trading)
        
        # Vérifier le compte
        account = self.alpaca.get_account()
        if account:
            self.capital = account['portfolio_value']
            logger.info(f"💰 Capital disponible: ${self.capital:,.2f}")
            logger.info(f"💵 Cash: ${account['cash']:,.2f}")
            logger.info(f"📊 Positions: ${self.capital - account['cash']:,.2f}")
        else:
            raise Exception("❌ Impossible de se connecter à Alpaca")
        
        # Paramètres de risque
        self.max_position_size = 0.1  # Max 10% du capital par position
        self.stop_loss_pct = 0.05     # Stop loss à -5%
        self.take_profit_pct = 0.15   # Take profit à +15%
        
        logger.info(f"🎯 Max position size: {self.max_position_size*100:.0f}%")
        logger.info(f"🛑 Stop Loss: {self.stop_loss_pct*100:.0f}%")
        logger.info(f"🎯 Take Profit: {self.take_profit_pct*100:.0f}%")
    
    def check_risk_management(self):
        """Vérifier stop loss et take profit sur les positions ouvertes"""
        positions = self.alpaca.get_positions()
        
        for pos in positions:
            symbol = pos['symbol']
            unrealized_plpc = pos['unrealized_plpc']
            
            # Stop Loss
            if unrealized_plpc <= -self.stop_loss_pct:
                logger.warning(f"🛑 STOP LOSS déclenché pour {symbol}: {unrealized_plpc*100:.2f}%")
                self.alpaca.close_position(symbol)
            
            # Take Profit
            elif unrealized_plpc >= self.take_profit_pct:
                logger.info(f"🎯 TAKE PROFIT déclenché pour {symbol}: {unrealized_plpc*100:.2f}%")
                self.alpaca.close_position(symbol)
    
    def execute_signals(self):
        """Exécuter les signaux du Brain AI"""
        logger.info("\n" + "="*70)
        logger.info(f"🧠 ANALYSE DES SIGNAUX - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*70)
        
        # Obtenir prédictions
        predictions = self.brain.predict_all()
        
        # Récupérer compte et positions actuelles
        account = self.alpaca.get_account()
        current_positions = {pos['symbol']: pos for pos in self.alpaca.get_positions()}
        
        actions_taken = {'buy': 0, 'sell': 0, 'hold': 0}
        
        for sector, sector_preds in predictions.items():
            logger.info(f"\n🧠 {sector.upper()}:")
            
            for pred in sector_preds:
                symbol = pred['ticker']
                action = pred['action']
                
                # Obtenir prix actuel
                current_price = self.alpaca.get_current_price(symbol)
                if current_price is None:
                    logger.warning(f"  ⚠️  {symbol}: Prix indisponible")
                    continue
                
                emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '⚪'}[action]
                logger.info(f"  {emoji} {symbol}: {action} @ ${current_price:.2f}")
                
                # SIGNAL BUY
                if action == 'BUY':
                    # Vérifier si on a déjà une position
                    if symbol in current_positions:
                        logger.info(f"     ⏭️  Position déjà ouverte sur {symbol}")
                        actions_taken['hold'] += 1
                        continue
                    
                    # Calculer la taille de la position
                    max_invest = self.capital * self.max_position_size
                    allocated = pred['capital']
                    invest_amount = min(max_invest, allocated, float(account['cash']))
                    
                    # Vérifier qu'on a assez de cash
                    if invest_amount < 100:  # Minimum $100
                        logger.warning(f"     ⚠️  Cash insuffisant: ${invest_amount:.2f}")
                        continue
                    
                    # Calculer le nombre d'actions
                    qty = int(invest_amount / current_price)
                    if qty < 1:
                        logger.warning(f"     ⚠️  Quantité trop faible: {qty}")
                        continue
                    
                    # Placer l'ordre
                    logger.info(f"     💰 Achat: {qty} actions x ${current_price:.2f} = ${qty*current_price:,.2f}")
                    order = self.alpaca.place_market_order(symbol, qty, 'buy')
                    
                    if order:
                        actions_taken['buy'] += 1
                        logger.info(f"     ✅ Ordre placé: {order['id']}")
                    else:
                        logger.error(f"     ❌ Échec de l'ordre")
                
                # SIGNAL SELL
                elif action == 'SELL':
                    # Vérifier si on a une position
                    if symbol not in current_positions:
                        logger.info(f"     ⏭️  Pas de position sur {symbol}")
                        actions_taken['hold'] += 1
                        continue
                    
                    # Fermer la position
                    pos = current_positions[symbol]
                    logger.info(f"     💰 Vente: {pos['qty']} actions @ ${current_price:.2f}")
                    logger.info(f"     📊 P&L: ${pos['unrealized_pl']:+,.2f} ({pos['unrealized_plpc']*100:+.2f}%)")
                    
                    if self.alpaca.close_position(symbol):
                        actions_taken['sell'] += 1
                        logger.info(f"     ✅ Position fermée")
                    else:
                        logger.error(f"     ❌ Échec de la fermeture")
                
                # SIGNAL HOLD
                else:
                    actions_taken['hold'] += 1
        
        # Résumé
        logger.info("\n" + "="*70)
        logger.info("📊 RÉSUMÉ DU CYCLE")
        logger.info("="*70)
        logger.info(f"🎯 Actions: {actions_taken['buy']} BUY | {actions_taken['sell']} SELL | {actions_taken['hold']} HOLD")
        
        # État du compte
        account = self.alpaca.get_account()
        logger.info(f"💰 Cash: ${account['cash']:,.2f}")
        logger.info(f"📈 Portfolio: ${account['portfolio_value']:,.2f}")
        logger.info(f"💵 P&L: ${account['equity'] - self.capital:+,.2f}")
    
    def run(self, check_interval_minutes=60):
        """
        Boucle principale de trading
        
        Args:
            check_interval_minutes: Intervalle entre chaque analyse
        """
        logger.info("\n" + "="*70)
        logger.info("🚀 DÉMARRAGE DU LIVE TRADER")
        logger.info("="*70)
        logger.info(f"⏱️  Intervalle: {check_interval_minutes} minutes")
        logger.info(f"📊 Mode: {'Paper Trading' if self.paper_trading else '🔴 LIVE TRADING'}")
        
        if not self.paper_trading:
            logger.warning("⚠️  MODE LIVE - TRADES RÉELS !")
            response = input("Continuer? (yes/no): ")
            if response.lower() != 'yes':
                logger.info("❌ Annulé")
                return
        
        cycle = 0
        
        try:
            while True:
                cycle += 1
                logger.info(f"\n📍 Cycle {cycle}")
                
                # 1. Risk management
                self.check_risk_management()
                
                # 2. Exécuter signaux
                self.execute_signals()
                
                # 3. Attendre
                logger.info(f"\n⏳ Prochain cycle dans {check_interval_minutes} minutes...")
                time.sleep(check_interval_minutes * 60)
        
        except KeyboardInterrupt:
            logger.info("\n\n🛑 Arrêt manuel")
        
        except Exception as e:
            logger.error(f"\n❌ Erreur: {e}", exc_info=True)
        
        finally:
            # Résumé final
            logger.info("\n" + "="*70)
            logger.info("📊 RÉSUMÉ FINAL")
            logger.info("="*70)
            
            account = self.alpaca.get_account()
            positions = self.alpaca.get_positions()
            
            logger.info(f"💰 Capital initial: ${self.capital:,.2f}")
            logger.info(f"💵 Portfolio final: ${account['portfolio_value']:,.2f}")
            logger.info(f"📈 P&L total: ${account['portfolio_value'] - self.capital:+,.2f}")
            logger.info(f"🔢 Positions ouvertes: {len(positions)}")
