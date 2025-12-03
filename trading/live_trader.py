# trading/live_trader.py - VERSION COMPLÈTE AVEC BDD

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
from datetime import datetime, date
import time

logger = setup_logging(__name__, 'live_trader.log')

# ========== INTÉGRATION BASE DE DONNÉES ==========
try:
    from database.db import log_prediction, save_daily_summary, get_trade_history
    DB_AVAILABLE = True
    logger.info("✅ Module database disponible")
except ImportError:
    DB_AVAILABLE = False
    logger.warning("⚠️  Module database non disponible")

class LiveTrader:
    """Trader live avec Alpaca - STRATÉGIE OPTIMISÉE + BDD"""
    
    def __init__(self, paper_trading=True, capital=None):
        self.paper_trading = paper_trading
        
        self.alpaca = AlpacaClient(paper_trading=paper_trading)
        self.brain = BrainTrader(capital=capital, paper_trading=paper_trading)
        
        account = self.alpaca.get_account()
        if account:
            self.initial_capital = account['portfolio_value']
            self.available_buying_power = float(account['buying_power'])
            
            logger.info(f"💰 Portfolio total: ${self.initial_capital:,.2f}")
            logger.info(f"💵 Buying Power: ${self.available_buying_power:,.2f}")
            logger.info(f"💸 Cash: ${float(account['cash']):,.2f}")
        else:
            raise Exception("❌ Impossible de se connecter à Alpaca")
        
        # Paramètres de risque
        self.max_position_size = 0.05      # 5% du portfolio
        self.min_position_size = 0.02      # 2% minimum
        self.stop_loss_pct = 0.05
        self.take_profit_pct = 0.15
        self.min_trade_amount = 100.0
        
        # ✅ Paramètres de renforcement
        self.max_position_accumulation = 0.10  # Max 10% du portfolio sur 1 ticker
        self.add_to_winner = True              # Renforcer les positions gagnantes
        self.add_to_loser = False              # Ne pas moyenner à la baisse
        
        logger.info(f"🎯 Position size: {self.min_position_size*100:.0f}%-{self.max_position_size*100:.0f}%")
        logger.info(f"📈 Max accumulation: {self.max_position_accumulation*100:.0f}%")
        logger.info(f"🛑 Stop Loss: {self.stop_loss_pct*100:.0f}%")
        logger.info(f"🎯 Take Profit: {self.take_profit_pct*100:.0f}%")
        logger.info(f"📊 BDD: {'✅ Activée' if DB_AVAILABLE else '❌ Non configurée'}")
    
    def check_risk_management(self):
        """Vérifier stop loss et take profit"""
        positions = self.alpaca.get_positions()
        
        for pos in positions:
            symbol = pos['symbol']
            unrealized_plpc = pos['unrealized_plpc']
            
            # Stop Loss
            if unrealized_plpc <= -self.stop_loss_pct:
                logger.warning(f"🛑 STOP LOSS: {symbol} ({unrealized_plpc*100:.2f}%)")
                self.alpaca.close_position(symbol, reason=f'Stop Loss {unrealized_plpc*100:.1f}%')
            
            # Take Profit
            elif unrealized_plpc >= self.take_profit_pct:
                logger.info(f"🎯 TAKE PROFIT: {symbol} ({unrealized_plpc*100:.2f}%)")
                self.alpaca.close_position(symbol, reason=f'Take Profit {unrealized_plpc*100:.1f}%')
    
    def should_add_to_position(self, symbol, position, current_price):
        """
        Décider si on doit renforcer une position existante
        
        Returns:
            bool, str: (should_add, reason)
        """
        account = self.alpaca.get_account()
        portfolio_value = account['portfolio_value']
        
        # Taille actuelle de la position en % du portfolio
        position_pct = position['market_value'] / portfolio_value
        
        # 1. Vérifier si on a atteint le max
        if position_pct >= self.max_position_accumulation:
            return False, f"Max accumulation atteint ({position_pct*100:.1f}%)"
        
        # 2. Vérifier le P&L
        unrealized_plpc = position['unrealized_plpc']
        
        # Position gagnante
        if unrealized_plpc > 0:
            if self.add_to_winner:
                return True, f"Renforcer gagnant (+{unrealized_plpc*100:.1f}%)"
            else:
                return False, "Mode renforcement gagnants désactivé"
        
        # Position perdante
        elif unrealized_plpc < 0:
            if self.add_to_loser:
                return True, f"Moyenner à la baisse ({unrealized_plpc*100:.1f}%)"
            else:
                return False, "Pas de moyenne à la baisse"
        
        # Position neutre
        else:
            return True, "Position neutre, renforcement OK"
    
    def execute_signals(self):
        """Exécuter les signaux du Brain AI - VERSION OPTIMISÉE"""
        logger.info("\n" + "="*70)
        logger.info(f"🧠 ANALYSE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*70)
        
        predictions = self.brain.predict_all()
        
        account = self.alpaca.get_account()
        available_buying_power = float(account['buying_power'])
        portfolio_value = account['portfolio_value']
        
        logger.info(f"💵 Buying Power: ${available_buying_power:,.2f}")
        logger.info(f"📊 Portfolio: ${portfolio_value:,.2f}")
        
        current_positions = {pos['symbol']: pos for pos in self.alpaca.get_positions()}
        
        actions = {'buy': 0, 'add': 0, 'sell': 0, 'hold': 0}
        
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
                
                # ✅ LOGGER LA PRÉDICTION DANS LA BDD
                if DB_AVAILABLE:
                    try:
                        log_prediction(
                            symbol=symbol,
                            sector=sector,
                            prediction=1 if action == 'BUY' else 0,
                            confidence=pred.get('confidence', 0.0),
                            action=action,
                            features={}
                        )
                    except Exception as e:
                        logger.error(f"❌ Erreur log prediction: {e}")
                
                # ✅ VÉRIFIER SI POSITION EXISTE
                position = current_positions.get(symbol)
                
                if position:
                    position_value = position['market_value']
                    position_pct = (position_value / portfolio_value) * 100
                    pl = position['unrealized_pl']
                    pl_pct = position['unrealized_plpc'] * 100
                    
                    logger.info(f"  {emoji} {symbol}: {action} @ ${current_price:.2f} | Position: ${position_value:,.0f} ({position_pct:.1f}%) | P&L: ${pl:+,.0f} ({pl_pct:+.1f}%)")
                else:
                    logger.info(f"  {emoji} {symbol}: {action} @ ${current_price:.2f} | Pas de position")
                
                # ===== SIGNAL BUY =====
                if action == 'BUY':
                    
                    # CAS 1 : Position existe déjà
                    if position:
                        should_add, reason = self.should_add_to_position(symbol, position, current_price)
                        
                        if should_add:
                            # ✅ RENFORCER LA POSITION
                            max_invest = portfolio_value * self.max_position_size
                            current_value = position['market_value']
                            remaining_capacity = max_invest - current_value
                            
                            if remaining_capacity < self.min_trade_amount:
                                logger.info(f"     ⏭️  Capacité restante insuffisante: ${remaining_capacity:.0f}")
                                actions['hold'] += 1
                                continue
                            
                            invest_amount = min(remaining_capacity, available_buying_power * 0.5)
                            
                            if invest_amount < self.min_trade_amount:
                                logger.info(f"     ⏭️  Budget insuffisant pour renforcer")
                                actions['hold'] += 1
                                continue
                            
                            qty = int(invest_amount / current_price)
                            if qty < 1:
                                logger.info(f"     ⏭️  Quantité insuffisante: {qty}")
                                actions['hold'] += 1
                                continue
                            
                            actual_cost = qty * current_price
                            
                            logger.info(f"     📈 RENFORCEMENT: +{qty} x ${current_price:.2f} = ${actual_cost:,.2f}")
                            logger.info(f"     💡 Raison: {reason}")
                            
                            order = self.alpaca.place_market_order(symbol, qty, 'buy', reason=f'Renforcement: {reason}')
                            
                            if order:
                                actions['add'] += 1
                                available_buying_power -= actual_cost
                                logger.info(f"     ✅ Ordre: {order['id']}")
                            else:
                                logger.error(f"     ❌ Échec ordre")
                        
                        else:
                            # ❌ NE PAS RENFORCER
                            logger.info(f"     ⏭️  {reason}")
                            actions['hold'] += 1
                    
                    # CAS 2 : Nouvelle position
                    else:
                        max_invest = portfolio_value * self.max_position_size
                        invest_amount = min(max_invest, available_buying_power * 0.5)
                        
                        if invest_amount < self.min_trade_amount:
                            logger.info(f"     ⚠️  Budget insuffisant: ${invest_amount:.2f}")
                            continue
                        
                        qty = int(invest_amount / current_price)
                        if qty < 1:
                            logger.info(f"     ⚠️  Quantité insuffisante: {qty}")
                            continue
                        
                        actual_cost = qty * current_price
                        
                        logger.info(f"     💰 NOUVELLE POSITION: {qty} x ${current_price:.2f} = ${actual_cost:,.2f}")
                        
                        order = self.alpaca.place_market_order(symbol, qty, 'buy', reason='Nouvelle position AI')
                        
                        if order:
                            actions['buy'] += 1
                            available_buying_power -= actual_cost
                            logger.info(f"     ✅ Ordre: {order['id']}")
                        else:
                            logger.error(f"     ❌ Échec ordre")
                
                # ===== SIGNAL SELL =====
                elif action == 'SELL':
                    if not position:
                        logger.info(f"     ⏭️  Pas de position à vendre")
                        actions['hold'] += 1
                        continue
                    
                    logger.info(f"     💰 FERMETURE: {position['qty']:.2f} @ ${current_price:.2f}")
                    logger.info(f"     📊 P&L: ${position['unrealized_pl']:+,.2f} ({position['unrealized_plpc']*100:+.2f}%)")
                    
                    if self.alpaca.close_position(symbol, reason='Signal SELL AI'):
                        actions['sell'] += 1
                        proceeds = position['qty'] * current_price
                        available_buying_power += proceeds
                        logger.info(f"     ✅ Position fermée")
                    else:
                        logger.error(f"     ❌ Échec fermeture")
                
                # ===== SIGNAL HOLD =====
                else:
                    if position:
                        logger.info(f"     ⏸️  Conservation de la position")
                    actions['hold'] += 1
        
        # Résumé
        logger.info("\n" + "="*70)
        logger.info("📊 RÉSUMÉ")
        logger.info("="*70)
        logger.info(f"🎯 {actions['buy']} NOUVEAU | {actions['add']} RENFORT | {actions['sell']} VENTE | {actions['hold']} HOLD")
        
        account = self.alpaca.get_account()
        current_value = account['portfolio_value']
        
        logger.info(f"💰 Portfolio: ${current_value:,.2f}")
        logger.info(f"💵 Buying Power: ${account['buying_power']:,.2f}")
        
        total_pl = current_value - self.initial_capital
        pl_pct = (total_pl / self.initial_capital) * 100
        logger.info(f"💸 P&L session: ${total_pl:+,.2f} ({pl_pct:+.2f}%)")
    
    def save_daily_stats(self):
        """Sauvegarder les statistiques quotidiennes dans la BDD"""
        if not DB_AVAILABLE:
            logger.warning("⚠️  BDD non disponible pour save stats")
            return
        
        try:
            account = self.alpaca.get_account()
            positions = self.alpaca.get_positions()
            
            # Compter les trades du jour
            trades_today = get_trade_history(days=1)
            
            total_pl = sum(p['unrealized_pl'] for p in positions)
            
            save_daily_summary(
                date=date.today(),
                portfolio_value=account['portfolio_value'],
                cash=account['cash'],
                buying_power=account['buying_power'],
                total_pl=total_pl,
                positions_count=len(positions),
                trades_count=len(trades_today)
            )
            
            # Logger aussi les positions
            self.alpaca.log_current_positions()
            
            logger.info("✅ Stats quotidiennes sauvegardées dans BDD")
            
        except Exception as e:
            logger.error(f"❌ Erreur save_daily_stats: {e}")
    
    def run(self, check_interval_minutes=60):
        """Boucle principale"""
        logger.info("\n" + "="*70)
        logger.info("🚀 LIVE TRADER - STRATÉGIE OPTIMISÉE + BDD")
        logger.info("="*70)
        logger.info(f"⏱️  Intervalle: {check_interval_minutes} min")
        logger.info(f"📊 Mode: {'Paper' if self.paper_trading else '🔴 LIVE'}")
        logger.info(f"📊 BDD: {'✅ Activée' if DB_AVAILABLE else '❌ Non configurée'}")
        
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
                
                # Vérifier stop loss / take profit
                self.check_risk_management()
                
                # Exécuter les signaux AI
                self.execute_signals()
                
                # Sauvegarder stats toutes les 4 heures
                if cycle % 4 == 0:
                    self.save_daily_stats()
                
                logger.info(f"\n⏳ Prochain cycle dans {check_interval_minutes} min...")
                time.sleep(check_interval_minutes * 60)
        
        except KeyboardInterrupt:
            logger.info("\n\n🛑 Arrêt manuel")
        except Exception as e:
            logger.error(f"\n❌ Erreur: {e}", exc_info=True)
        finally:
            # Stats finales
            account = self.alpaca.get_account()
            final_value = account['portfolio_value']
            total_pl = final_value - self.initial_capital
            
            logger.info("\n" + "="*70)
            logger.info("📊 RÉSUMÉ FINAL")
            logger.info("="*70)
            logger.info(f"💰 Portfolio initial: ${self.initial_capital:,.2f}")
            logger.info(f"💵 Portfolio final: ${final_value:,.2f}")
            logger.info(f"📈 P&L total: ${total_pl:+,.2f} ({(total_pl/self.initial_capital)*100:+.2f}%)")
            
            # Dernière sauvegarde BDD
            if DB_AVAILABLE:
                self.save_daily_stats()