# trading/live_trader.py - VERSION COMPLÈTE AVEC ALERTES

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
    from database.db import log_prediction, save_daily_summary, get_trade_history, get_win_loss_ratio
    DB_AVAILABLE = True
    logger.info("✅ Module database disponible")
except ImportError:
    DB_AVAILABLE = False
    logger.warning("⚠️  Module database non disponible")

# ========== INTÉGRATION ALERTES ==========
try:
    from core.alerts import (
        send_alert, alert_trade, alert_profit, alert_loss,
        alert_daily_summary, alert_performance_warning,
        alert_startup, alert_shutdown
    )
    ALERTS_AVAILABLE = True
    logger.info("✅ Module alertes disponible")
except ImportError:
    ALERTS_AVAILABLE = False
    logger.warning("⚠️  Module alertes non disponible")

class LiveTrader:
    """Trader live avec Alpaca - STRATÉGIE OPTIMISÉE + BDD + ALERTES"""
    
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
        self.max_position_size = 0.05
        self.min_position_size = 0.02
        self.stop_loss_pct = 0.05
        self.take_profit_pct = 0.15
        self.min_trade_amount = 100.0
        
        # Paramètres de renforcement
        self.max_position_accumulation = 0.10
        self.add_to_winner = True
        self.add_to_loser = False
        
        logger.info(f"🎯 Position size: {self.min_position_size*100:.0f}%-{self.max_position_size*100:.0f}%")
        logger.info(f"📈 Max accumulation: {self.max_position_accumulation*100:.0f}%")
        logger.info(f"🛑 Stop Loss: {self.stop_loss_pct*100:.0f}%")
        logger.info(f"🎯 Take Profit: {self.take_profit_pct*100:.0f}%")
        logger.info(f"📊 BDD: {'✅ Activée' if DB_AVAILABLE else '❌ Non configurée'}")
        logger.info(f"🔔 Alertes: {'✅ Activées' if ALERTS_AVAILABLE else '❌ Non configurées'}")
    
    def check_risk_management(self):
        """Vérifier stop loss et take profit"""
        positions = self.alpaca.get_positions()
        
        for pos in positions:
            symbol = pos['symbol']
            unrealized_plpc = pos['unrealized_plpc']
            unrealized_pl = pos['unrealized_pl']
            
            # Stop Loss
            if unrealized_plpc <= -self.stop_loss_pct:
                logger.warning(f"🛑 STOP LOSS: {symbol} ({unrealized_plpc*100:.2f}%)")
                
                if self.alpaca.close_position(symbol, reason=f'Stop Loss {unrealized_plpc*100:.1f}%'):
                    # Alerte perte
                    if ALERTS_AVAILABLE:
                        alert_loss(symbol, unrealized_pl, unrealized_plpc * 100)
            
            # Take Profit
            elif unrealized_plpc >= self.take_profit_pct:
                logger.info(f"🎯 TAKE PROFIT: {symbol} ({unrealized_plpc*100:.2f}%)")
                
                if self.alpaca.close_position(symbol, reason=f'Take Profit {unrealized_plpc*100:.1f}%'):
                    # Alerte profit
                    if ALERTS_AVAILABLE:
                        alert_profit(symbol, unrealized_pl, unrealized_plpc * 100)
    
    def should_add_to_position(self, symbol, position, current_price):
        """Décider si on doit renforcer une position existante"""
        account = self.alpaca.get_account()
        portfolio_value = account['portfolio_value']
        
        position_pct = position['market_value'] / portfolio_value
        
        if position_pct >= self.max_position_accumulation:
            return False, f"Max accumulation atteint ({position_pct*100:.1f}%)"
        
        unrealized_plpc = position['unrealized_plpc']
        
        if unrealized_plpc > 0:
            if self.add_to_winner:
                return True, f"Renforcer gagnant (+{unrealized_plpc*100:.1f}%)"
            else:
                return False, "Mode renforcement gagnants désactivé"
        
        elif unrealized_plpc < 0:
            if self.add_to_loser:
                return True, f"Moyenner à la baisse ({unrealized_plpc*100:.1f}%)"
            else:
                return False, "Pas de moyenne à la baisse"
        
        else:
            return True, "Position neutre, renforcement OK"
    
    def execute_signals(self):
        """Exécuter les signaux du Brain AI"""
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
                
                # Logger la prédiction dans la BDD
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
                    
                    if position:
                        should_add, reason = self.should_add_to_position(symbol, position, current_price)
                        
                        if should_add:
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
                                
                                # Alerte trade
                                if ALERTS_AVAILABLE:
                                    alert_trade(symbol, 'BUY (Renforcement)', qty, current_price, actual_cost)
                            else:
                                logger.error(f"     ❌ Échec ordre")
                        
                        else:
                            logger.info(f"     ⏭️  {reason}")
                            actions['hold'] += 1
                    
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
                            
                            # Alerte trade
                            if ALERTS_AVAILABLE:
                                alert_trade(symbol, 'BUY', qty, current_price, actual_cost)
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
                        
                        # Alerte selon P&L
                        if ALERTS_AVAILABLE:
                            if position['unrealized_pl'] > 0:
                                alert_profit(symbol, position['unrealized_pl'], position['unrealized_plpc'] * 100)
                            else:
                                alert_loss(symbol, position['unrealized_pl'], position['unrealized_plpc'] * 100)
                    else:
                        logger.error(f"     ❌ Échec fermeture")
                
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
        """Sauvegarder les statistiques quotidiennes"""
        if not DB_AVAILABLE:
            logger.warning("⚠️  BDD non disponible pour save stats")
            return
        
        try:
            account = self.alpaca.get_account()
            positions = self.alpaca.get_positions()
            
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
            
            self.alpaca.log_current_positions()
            
            logger.info("✅ Stats quotidiennes sauvegardées dans BDD")
            
            # Alerte résumé quotidien
            if ALERTS_AVAILABLE:
                pl_pct = (total_pl / self.initial_capital * 100) if self.initial_capital > 0 else 0
                alert_daily_summary(
                    portfolio_value=account['portfolio_value'],
                    pl=total_pl,
                    pl_pct=pl_pct,
                    trades_count=len(trades_today)
                )
            
        except Exception as e:
            logger.error(f"❌ Erreur save_daily_stats: {e}")
    
    def check_performance_alerts(self):
        """Vérifier et envoyer alertes de performance"""
        if not ALERTS_AVAILABLE or not DB_AVAILABLE:
            return
        
        try:
            # Vérifier win rate
            win_loss = get_win_loss_ratio(days=7)
            if win_loss['win_rate'] < 50 and win_loss['total'] > 10:
                alert_performance_warning(win_loss['win_rate'], 7)
        
        except Exception as e:
            logger.error(f"❌ Erreur check_performance_alerts: {e}")
    
    def run(self, check_interval_minutes=60):
        """Boucle principale"""
        logger.info("\n" + "="*70)
        logger.info("🚀 LIVE TRADER - STRATÉGIE OPTIMISÉE + BDD + ALERTES")
        logger.info("="*70)
        logger.info(f"⏱️  Intervalle: {check_interval_minutes} min")
        logger.info(f"📊 Mode: {'Paper' if self.paper_trading else '🔴 LIVE'}")
        logger.info(f"📊 BDD: {'✅ Activée' if DB_AVAILABLE else '❌ Non configurée'}")
        logger.info(f"🔔 Alertes: {'✅ Activées' if ALERTS_AVAILABLE else '❌ Non configurées'}")
        
        if not self.paper_trading:
            logger.warning("⚠️  MODE LIVE - REAL MONEY!")
            response = input("Continuer? (yes/no): ")
            if response.lower() != 'yes':
                return
        
        # Alerte démarrage
        if ALERTS_AVAILABLE:
            alert_startup()
        
        cycle = 0
        
        try:
            while True:
                cycle += 1
                logger.info(f"\n📍 Cycle {cycle}")
                
                self.check_risk_management()
                self.execute_signals()
                
                # Sauvegarder stats toutes les 4 heures
                if cycle % 4 == 0:
                    self.save_daily_stats()
                
                # Vérifier alertes performance toutes les 12 heures
                if cycle % 12 == 0:
                    self.check_performance_alerts()
                
                logger.info(f"\n⏳ Prochain cycle dans {check_interval_minutes} min...")
                time.sleep(check_interval_minutes * 60)
        
        except KeyboardInterrupt:
            logger.info("\n\n🛑 Arrêt manuel")
        except Exception as e:
            logger.error(f"\n❌ Erreur: {e}", exc_info=True)
            
            # Alerte erreur critique
            if ALERTS_AVAILABLE:
                send_alert(
                    f"🚨 **ERREUR CRITIQUE**\n\n"
                    f"Le bot s'est arrêté\n"
                    f"Erreur: {str(e)[:200]}",
                    priority='ERROR'
                )
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
            
            # Alerte arrêt
            if ALERTS_AVAILABLE:
                alert_shutdown(final_value, total_pl)
            
            # Dernière sauvegarde BDD
            if DB_AVAILABLE:
                self.save_daily_stats()