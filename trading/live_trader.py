# trading/live_trader.py
"""Trader en temps réel avec Alpaca - ACTIONS UNIQUEMENT"""

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
    """Trader live avec Alpaca - Actions uniquement"""
    
    def __init__(self, paper_trading=True, capital=None):
        self.paper_trading = paper_trading
        
        # Client Alpaca
        self.alpaca = AlpacaClient(paper_trading=paper_trading)
        self.brain = BrainTrader(capital=capital, paper_trading=paper_trading)
        
        # Vérifier le compte
        account = self.alpaca.get_account()
        if account:
            self.initial_capital = account['portfolio_value']
            self.available_buying_power = float(account['buying_power'])  # ✅ BUYING POWER
            
            logger.info(f"💰 Portfolio total: ${self.initial_capital:,.2f}")
            logger.info(f"💵 Buying Power: ${self.available_buying_power:,.2f}")
            logger.info(f"💸 Cash: ${float(account['cash']):,.2f}")
            
            positions_value = self.initial_capital - float(account['cash'])
            logger.info(f"📊 Positions ouvertes: ${positions_value:,.2f}")
        else:
            raise Exception("❌ Impossible de se connecter à Alpaca")
        
        # Paramètres de risque
        self.max_position_size = 0.05      # ✅ 5% du buying power (plus conservateur)
        self.stop_loss_pct = 0.05
        self.take_profit_pct = 0.15
        self.min_trade_amount = 100.0
        
        logger.info(f"🎯 Max position: {self.max_position_size*100:.0f}% du buying power")
        logger.info(f"🛑 Stop Loss: {self.stop_loss_pct*100:.0f}%")
        logger.info(f"🎯 Take Profit: {self.take_profit_pct*100:.0f}%")
    
    def check_risk_management(self):
        """Vérifier stop loss et take profit"""
        positions = self.alpaca.get_positions()
        
        for pos in positions:
            symbol = pos['symbol']
            unrealized_plpc = pos['unrealized_plpc']
            
            # Stop Loss
            if unrealized_plpc <= -self.stop_loss_pct:
                logger.warning(f"🛑 STOP LOSS: {symbol} ({unrealized_plpc*100:.2f}%)")
                self.alpaca.close_position(symbol)
            
            # Take Profit
            elif unrealized_plpc >= self.take_profit_pct:
                logger.info(f"🎯 TAKE PROFIT: {symbol} ({unrealized_plpc*100:.2f}%)")
                self.alpaca.close_position(symbol)
    
    def execute_signals(self):
        """Exécuter les signaux du Brain AI"""
        logger.info("\n" + "="*70)
        logger.info(f"🧠 ANALYSE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*70)
        
        predictions = self.brain.predict_all()
        
        # ✅ UTILISER BUYING POWER
        account = self.alpaca.get_account()
        available_buying_power = float(account['buying_power'])
        
        logger.info(f"💵 Buying Power disponible: ${available_buying_power:,.2f}")
        
        current_positions = {pos['symbol']: pos for pos in self.alpaca.get_positions()}
        
        actions = {'buy': 0, 'sell': 0, 'hold': 0}
        
        # Collecter signaux BUY
        buy_signals = []
        
        for sector, sector_preds in predictions.items():
            for pred in sector_preds:
                if pred['action'] == 'BUY':
                    symbol = pred['ticker']
                    
                    if symbol not in current_positions:
                        price = self.alpaca.get_current_price(symbol)
                        if price:
                            buy_signals.append({
                                'symbol': symbol,
                                'price': price,
                                'sector': sector
                            })
        
        # Calculer budget par signal
        if buy_signals:
            # ✅ Utiliser max_position_size du BUYING POWER
            max_per_position = available_buying_power * self.max_position_size
            
            logger.info(f"🎯 {len(buy_signals)} signaux BUY détectés")
            logger.info(f"💰 Budget max par position: ${max_per_position:,.2f}")
        
        # EXÉCUTER LES TRADES
        for sector, sector_preds in predictions.items():
            logger.info(f"\n🧠 {sector.upper()}:")
            
            for pred in sector_preds:
                symbol = pred['ticker']
                action = pred['action']
                
                current_price = self.alpaca.get_current_price(symbol)
                if current_price is None:
                    logger.warning(f"  ⚠️  {symbol}: Prix indisponible")
                    continue
                
                emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '⚪'}[action]
                logger.info(f"  {emoji} {symbol}: {action} @ ${current_price:.2f}")
                
                # SIGNAL BUY
                if action == 'BUY':
                    if symbol in current_positions:
                        logger.info(f"     ⏭️  Position déjà ouverte")
                        actions['hold'] += 1
                        continue
                    
                    # ✅ Utiliser max_position_size
                    max_invest = available_buying_power * self.max_position_size
                    invest_amount = min(max_invest, available_buying_power)
                    
                    if invest_amount < self.min_trade_amount:
                        logger.warning(f"     ⚠️  Budget insuffisant: ${invest_amount:.2f}")
                        continue
                    
                    # Calculer quantité
                    qty = int(invest_amount / current_price)
                    if qty < 1:
                        logger.warning(f"     ⚠️  Quantité insuffisante: {qty}")
                        continue
                    
                    actual_cost = qty * current_price
                    
                    # Vérifier buying power
                    if actual_cost > available_buying_power:
                        logger.warning(f"     ⚠️  Coût ${actual_cost:.2f} > BP ${available_buying_power:.2f}")
                        continue
                    
                    # Placer ordre
                    logger.info(f"     💰 Achat: {qty} x ${current_price:.2f} = ${actual_cost:,.2f}")
                    order = self.alpaca.place_market_order(symbol, qty, 'buy')
                    
                    if order:
                        actions['buy'] += 1
                        available_buying_power -= actual_cost
                        logger.info(f"     ✅ Ordre: {order['id']}")
                        logger.info(f"     💵 BP restant: ${available_buying_power:,.2f}")
                    else:
                        logger.error(f"     ❌ Échec ordre")
                
                # SIGNAL SELL
                elif action == 'SELL':
                    if symbol not in current_positions:
                        logger.info(f"     ⏭️  Pas de position")
                        actions['hold'] += 1
                        continue
                    
                    pos = current_positions[symbol]
                    logger.info(f"     💰 Vente: {pos['qty']:.2f} @ ${current_price:.2f}")
                    logger.info(f"     📊 P&L: ${pos['unrealized_pl']:+,.2f} ({pos['unrealized_plpc']*100:+.2f}%)")
                    
                    if self.alpaca.close_position(symbol):
                        actions['sell'] += 1
                        proceeds = pos['qty'] * current_price
                        available_buying_power += proceeds
                        logger.info(f"     ✅ Position fermée")
                        logger.info(f"     💵 BP après vente: ${available_buying_power:,.2f}")
                    else:
                        logger.error(f"     ❌ Échec fermeture")
                
                else:
                    actions['hold'] += 1
        
        # Résumé
        logger.info("\n" + "="*70)
        logger.info("📊 RÉSUMÉ")
        logger.info("="*70)
        logger.info(f"🎯 {actions['buy']} BUY | {actions['sell']} SELL | {actions['hold']} HOLD")
        
        account = self.alpaca.get_account()
        current_value = account['portfolio_value']
        
        logger.info(f"💰 Portfolio: ${current_value:,.2f}")
        logger.info(f"💵 Buying Power: ${account['buying_power']:,.2f}")
        
        total_pl = current_value - self.initial_capital
        pl_pct = (total_pl / self.initial_capital) * 100
        logger.info(f"💸 P&L session: ${total_pl:+,.2f} ({pl_pct:+.2f}%)")
    
    def run(self, check_interval_minutes=60):
        """Boucle principale"""
        logger.info("\n" + "="*70)
        logger.info("🚀 LIVE TRADER - ACTIONS UNIQUEMENT")
        logger.info("="*70)
        logger.info(f"⏱️  Intervalle: {check_interval_minutes} min")
        logger.info(f"📊 Mode: {'Paper' if self.paper_trading else '🔴 LIVE'}")
        
        if not self.paper_trading:
            logger.warning("⚠️  MODE LIVE - REAL MONEY!")
            response = input("Continuer? (yes/no): ")
            if response.lower() != 'yes':
                return
        
        cycle = 0
        
        try:
            while True:
                cycle += 1
                logger.info(f"\n📍 Cycle {cycle}")
                
                self.check_risk_management()
                self.execute_signals()
                
                logger.info(f"\n⏳ Prochain cycle dans {check_interval_minutes} min...")
                time.sleep(check_interval_minutes * 60)
        
        except KeyboardInterrupt:
            logger.info("\n\n🛑 Arrêt manuel")
        except Exception as e:
            logger.error(f"\n❌ Erreur: {e}", exc_info=True)
        finally:
            account = self.alpaca.get_account()
            final_value = account['portfolio_value']
            total_pl = final_value - self.initial_capital
            
            logger.info("\n" + "="*70)
            logger.info("📊 RÉSUMÉ FINAL")
            logger.info("="*70)
            logger.info(f"💰 Portfolio initial: ${self.initial_capital:,.2f}")
            logger.info(f"💵 Portfolio final: ${final_value:,.2f}")
            logger.info(f"📈 P&L total: ${total_pl:+,.2f} ({(total_pl/self.initial_capital)*100:+.2f}%)")