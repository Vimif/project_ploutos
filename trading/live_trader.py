# trading/live_trader.py - VERSION FINALE AVEC MONITORING

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

# ========== INTÉGRATIONS ==========
try:
    from database.db import log_prediction, save_daily_summary, get_trade_history, get_win_loss_ratio
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

try:
    from core.alerts import (
        send_alert, alert_trade, alert_profit, alert_loss,
        alert_daily_summary, alert_performance_warning,
        alert_startup, alert_shutdown
    )
    ALERTS_AVAILABLE = True
except ImportError:
    ALERTS_AVAILABLE = False

try:
    from core.risk_manager import RiskManager
    RISK_AVAILABLE = True
except ImportError:
    RISK_AVAILABLE = False

try:
    from core.monitoring import start_monitoring, get_metrics
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

class LiveTrader:
    """Trader live avec Alpaca - VERSION FINALE"""
    
    def __init__(self, paper_trading=True, capital=None, monitoring_port=9090):
        self.paper_trading = paper_trading
        
        # ✅ DÉMARRER MONITORING EN PREMIER
        self.metrics = None
        if MONITORING_AVAILABLE:
            try:
                self.metrics = start_monitoring(port=monitoring_port)
                logger.info(f"✅ Monitoring démarré: http://localhost:{monitoring_port}/metrics")
            except Exception as e:
                logger.error(f"❌ Erreur monitoring: {e}")
        
        self.alpaca = AlpacaClient(paper_trading=paper_trading)
        self.brain = BrainTrader(capital=capital, paper_trading=paper_trading)
        
        account = self.alpaca.get_account()
        if account:
            self.initial_capital = account['portfolio_value']
            self.available_buying_power = float(account['buying_power'])
            
            logger.info(f"💰 Portfolio total: ${self.initial_capital:,.2f}")
            logger.info(f"💵 Buying Power: ${self.available_buying_power:,.2f}")
            logger.info(f"💸 Cash: ${float(account['cash']):,.2f}")
            
            # ✅ INITIALISER MÉTRIQUES
            if self.metrics:
                positions = self.alpaca.get_positions()
                self.metrics.update_portfolio_metrics(account, positions)
                self.metrics.update_daily_metrics(self.initial_capital, account['portfolio_value'])
        else:
            raise Exception("❌ Impossible de se connecter à Alpaca")
        
        # Risk Manager
        if RISK_AVAILABLE:
            self.risk_manager = RiskManager(
                max_portfolio_risk=0.01,
                max_daily_loss=0.03,
                max_position_size=0.05,
                max_correlation=0.7
            )
            self.risk_manager.reset_daily_stats(self.initial_capital)
        else:
            self.risk_manager = None
        
        # Paramètres
        self.stop_loss_pct = 0.02
        self.take_profit_pct = 0.15
        self.min_trade_amount = 100.0
        self.max_position_accumulation = 0.10
        self.add_to_winner = True
        self.add_to_loser = False
        
        logger.info(f"🛑 Stop Loss: {self.stop_loss_pct*100:.0f}%")
        logger.info(f"🎯 Take Profit: {self.take_profit_pct*100:.0f}%")
        logger.info(f"📊 BDD: {'✅' if DB_AVAILABLE else '❌'}")
        logger.info(f"🔔 Alertes: {'✅' if ALERTS_AVAILABLE else '❌'}")
        logger.info(f"🛡️ Risk Management: {'✅' if RISK_AVAILABLE else '❌'}")
        logger.info(f"📈 Monitoring: {'✅' if MONITORING_AVAILABLE else '❌'}")
    
    def check_risk_management(self):
        """Vérifier stop loss, take profit et risque"""
        positions = self.alpaca.get_positions()
        account = self.alpaca.get_account()
        
        # Circuit breaker
        if RISK_AVAILABLE:
            if not self.risk_manager.check_daily_loss_limit(account['portfolio_value']):
                logger.error("🚨 Circuit breaker actif")
                
                # ✅ MÉTRIQUES
                if self.metrics:
                    self.metrics.update_risk_metrics(circuit_breaker_active=True, risky_positions=len(positions))
                
                if ALERTS_AVAILABLE:
                    send_alert(
                        f"🚨 **CIRCUIT BREAKER ACTIVÉ**\n\n"
                        f"Perte > {self.risk_manager.max_daily_loss*100:.0f}%\n"
                        "Fermeture positions",
                        priority='ERROR'
                    )
                
                for pos in positions:
                    self.alpaca.close_position(pos['symbol'], reason='Circuit Breaker')
                
                return
            
            # Rapport risque
            if len(positions) > 0:
                risk_report = self.risk_manager.get_risk_report(positions, account['portfolio_value'])
                
                # ✅ MÉTRIQUES RISQUE
                if self.metrics:
                    self.metrics.update_risk_metrics(
                        circuit_breaker_active=False,
                        risky_positions=risk_report['risky_positions_count']
                    )
                
                if risk_report['risky_positions_count'] > 0:
                    self.risk_manager.print_risk_summary(risk_report)
                    
                    if risk_report['risky_positions_count'] >= 3 and ALERTS_AVAILABLE:
                        send_alert(
                            f"⚠️ **{risk_report['risky_positions_count']} POSITIONS À RISQUE**",
                            priority='WARNING'
                        )
        
        # Stop Loss / Take Profit
        for pos in positions:
            symbol = pos['symbol']
            unrealized_plpc = pos['unrealized_plpc']
            unrealized_pl = pos['unrealized_pl']
            
            if unrealized_plpc <= -self.stop_loss_pct:
                logger.warning(f"🛑 STOP LOSS: {symbol} ({unrealized_plpc*100:.2f}%)")
                
                if self.alpaca.close_position(symbol, reason=f'Stop Loss {unrealized_plpc*100:.1f}%'):
                    # ✅ MÉTRIQUES
                    if self.metrics:
                        self.metrics.record_trade(symbol, 'SELL', abs(unrealized_pl), result='loss')
                    
                    if RISK_AVAILABLE:
                        self.risk_manager.log_trade(symbol, 'SELL', unrealized_pl)
                    
                    if ALERTS_AVAILABLE:
                        alert_loss(symbol, unrealized_pl, unrealized_plpc * 100)
            
            elif unrealized_plpc >= self.take_profit_pct:
                logger.info(f"🎯 TAKE PROFIT: {symbol} ({unrealized_plpc*100:.2f}%)")
                
                if self.alpaca.close_position(symbol, reason=f'Take Profit {unrealized_plpc*100:.1f}%'):
                    # ✅ MÉTRIQUES
                    if self.metrics:
                        self.metrics.record_trade(symbol, 'SELL', unrealized_pl, result='win')
                    
                    if RISK_AVAILABLE:
                        self.risk_manager.log_trade(symbol, 'SELL', unrealized_pl)
                    
                    if ALERTS_AVAILABLE:
                        alert_profit(symbol, unrealized_pl, unrealized_plpc * 100)
    
    def calculate_position_size_with_risk(self, symbol: str, current_price: float, portfolio_value: float) -> int:
        """Calculer taille position avec risk management"""
        
        if not RISK_AVAILABLE or self.risk_manager is None:
            max_invest = portfolio_value * 0.05
            return int(max_invest / current_price)
        
        quantity, position_value = self.risk_manager.calculate_position_size(
            portfolio_value=portfolio_value,
            entry_price=current_price,
            stop_loss_pct=self.stop_loss_pct,
            risk_pct=None
        )
        
        return quantity
    
    def should_add_to_position(self, symbol, position, current_price):
        """Décider si renforcement"""
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
        
        current_positions = self.alpaca.get_positions()
        
        # ✅ METTRE À JOUR MÉTRIQUES PORTFOLIO
        if self.metrics:
            self.metrics.update_portfolio_metrics(account, current_positions)
            self.metrics.update_daily_metrics(self.initial_capital, portfolio_value)
        
        if RISK_AVAILABLE and len(current_positions) > 0:
            should_reduce, reason = self.risk_manager.should_reduce_exposure(current_positions, portfolio_value)
            if should_reduce:
                logger.warning(f"⚠️  EXPOSITION ÉLEVÉE: {reason}")
        
        current_positions_dict = {pos['symbol']: pos for pos in current_positions}
        
        actions = {'buy': 0, 'add': 0, 'sell': 0, 'hold': 0}
        
        for sector, sector_preds in predictions.items():
            logger.info(f"\n🧠 {sector.upper()}:")
            
            for pred in sector_preds:
                symbol = pred['ticker']
                action = pred['action']
                
                # ✅ MÉTRIQUES PRÉDICTION
                if self.metrics:
                    self.metrics.record_prediction(action, sector)
                
                current_price = self.alpaca.get_current_price(symbol)
                if current_price is None:
                    logger.warning(f"  ⚠️  {symbol}: Prix indisponible")
                    if self.metrics:
                        self.metrics.record_error('price_fetch', 'unavailable')
                    continue
                
                emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '⚪'}[action]
                
                if DB_AVAILABLE:
                    try:
                        log_prediction(symbol, sector, 1 if action == 'BUY' else 0, 
                                     pred.get('confidence', 0.0), action, {})
                    except Exception as e:
                        logger.error(f"❌ Erreur log prediction: {e}")
                        if self.metrics:
                            self.metrics.record_error('database', 'log_prediction')
                
                position = current_positions_dict.get(symbol)
                
                if position:
                    position_value = position['market_value']
                    position_pct = (position_value / portfolio_value) * 100
                    pl = position['unrealized_pl']
                    pl_pct = position['unrealized_plpc'] * 100
                    
                    logger.info(f"  {emoji} {symbol}: {action} @ ${current_price:.2f} | "
                              f"Position: ${position_value:,.0f} ({position_pct:.1f}%) | "
                              f"P&L: ${pl:+,.0f} ({pl_pct:+.1f}%)")
                else:
                    logger.info(f"  {emoji} {symbol}: {action} @ ${current_price:.2f} | Pas de position")
                
                # ===== SIGNAL BUY =====
                if action == 'BUY':
                    start_time = time.time()
                    
                    if position:
                        should_add, reason = self.should_add_to_position(symbol, position, current_price)
                        
                        if should_add:
                            qty = self.calculate_position_size_with_risk(symbol, current_price, portfolio_value)
                            
                            max_position_value = portfolio_value * self.max_position_accumulation
                            remaining_capacity = max_position_value - position['market_value']
                            
                            if qty * current_price > remaining_capacity:
                                qty = int(remaining_capacity / current_price)
                            
                            if qty < 1:
                                logger.info(f"     ⏭️  Quantité insuffisante")
                                actions['hold'] += 1
                                continue
                            
                            actual_cost = qty * current_price
                            
                            if actual_cost < self.min_trade_amount:
                                logger.info(f"     ⏭️  Montant < minimum")
                                actions['hold'] += 1
                                continue
                            
                            logger.info(f"     📈 RENFORCEMENT: +{qty} x ${current_price:.2f} = ${actual_cost:,.2f}")
                            
                            order = self.alpaca.place_market_order(symbol, qty, 'buy', reason=f'Renforcement')
                            
                            if order:
                                actions['add'] += 1
                                available_buying_power -= actual_cost
                                execution_time = time.time() - start_time
                                
                                # ✅ MÉTRIQUES
                                if self.metrics:
                                    self.metrics.record_trade(symbol, 'BUY', actual_cost, execution_time, 'success')
                                
                                if RISK_AVAILABLE:
                                    self.risk_manager.log_trade(symbol, 'BUY')
                                
                                if ALERTS_AVAILABLE:
                                    alert_trade(symbol, 'BUY (Renforcement)', qty, current_price, actual_cost)
                                
                                logger.info(f"     ✅ Ordre: {order['id']}")
                            else:
                                # ✅ MÉTRIQUES ÉCHEC
                                if self.metrics:
                                    self.metrics.record_trade(symbol, 'BUY', actual_cost, 0, 'failed')
                                    self.metrics.record_error('trading', 'order_failed')
                                
                                logger.error(f"     ❌ Échec ordre")
                        else:
                            logger.info(f"     ⏭️  {reason}")
                            actions['hold'] += 1
                    
                    else:
                        # Nouvelle position
                        qty = self.calculate_position_size_with_risk(symbol, current_price, portfolio_value)
                        
                        if qty < 1:
                            logger.info(f"     ⚠️  Quantité insuffisante")
                            continue
                        
                        actual_cost = qty * current_price
                        
                        if actual_cost < self.min_trade_amount or actual_cost > available_buying_power:
                            logger.info(f"     ⚠️  Budget inadapté")
                            continue
                        
                        logger.info(f"     💰 NOUVELLE POSITION: {qty} x ${current_price:.2f} = ${actual_cost:,.2f}")
                        
                        order = self.alpaca.place_market_order(symbol, qty, 'buy', reason='Nouvelle position')
                        
                        if order:
                            actions['buy'] += 1
                            available_buying_power -= actual_cost
                            execution_time = time.time() - start_time
                            
                            # ✅ MÉTRIQUES
                            if self.metrics:
                                self.metrics.record_trade(symbol, 'BUY', actual_cost, execution_time, 'success')
                            
                            if RISK_AVAILABLE:
                                self.risk_manager.log_trade(symbol, 'BUY')
                            
                            if ALERTS_AVAILABLE:
                                alert_trade(symbol, 'BUY', qty, current_price, actual_cost)
                            
                            logger.info(f"     ✅ Ordre: {order['id']}")
                        else:
                            if self.metrics:
                                self.metrics.record_trade(symbol, 'BUY', actual_cost, 0, 'failed')
                                self.metrics.record_error('trading', 'order_failed')
                            
                            logger.error(f"     ❌ Échec ordre")
                
                # ===== SIGNAL SELL =====
                elif action == 'SELL':
                    if not position:
                        logger.info(f"     ⏭️  Pas de position à vendre")
                        actions['hold'] += 1
                        continue
                    
                    logger.info(f"     💰 FERMETURE: {position['qty']:.2f} @ ${current_price:.2f}")
                    logger.info(f"     📊 P&L: ${position['unrealized_pl']:+,.2f} ({position['unrealized_plpc']*100:+.2f}%)")
                    
                    start_time = time.time()
                    
                    if self.alpaca.close_position(symbol, reason='Signal SELL AI'):
                        actions['sell'] += 1
                        proceeds = position['qty'] * current_price
                        available_buying_power += proceeds
                        execution_time = time.time() - start_time
                        
                        result = 'win' if position['unrealized_pl'] > 0 else 'loss'
                        
                        # ✅ MÉTRIQUES
                        if self.metrics:
                            self.metrics.record_trade(symbol, 'SELL', proceeds, execution_time, result)
                        
                        if RISK_AVAILABLE:
                            self.risk_manager.log_trade(symbol, 'SELL', position['unrealized_pl'])
                        
                        if ALERTS_AVAILABLE:
                            if position['unrealized_pl'] > 0:
                                alert_profit(symbol, position['unrealized_pl'], position['unrealized_plpc'] * 100)
                            else:
                                alert_loss(symbol, position['unrealized_pl'], position['unrealized_plpc'] * 100)
                        
                        logger.info(f"     ✅ Position fermée")
                    else:
                        if self.metrics:
                            self.metrics.record_error('trading', 'close_failed')
                        
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
        
        # ✅ METTRE À JOUR MÉTRIQUES FINALES
        if self.metrics:
            self.metrics.total_pnl.set(total_pl)
            current_positions = self.alpaca.get_positions()
            self.metrics.update_portfolio_metrics(account, current_positions)
    
    def save_daily_stats(self):
        """Sauvegarder stats quotidiennes"""
        if not DB_AVAILABLE:
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
            logger.info("✅ Stats quotidiennes sauvegardées")
            
            # ✅ MÉTRIQUES PERFORMANCE
            if self.metrics and DB_AVAILABLE:
                win_loss = get_win_loss_ratio(days=30)
                if win_loss['total'] > 0:
                    self.metrics.update_performance_metrics(win_loss['win_rate'])
            
            if ALERTS_AVAILABLE:
                pl_pct = (total_pl / self.initial_capital * 100) if self.initial_capital > 0 else 0
                alert_daily_summary(account['portfolio_value'], total_pl, pl_pct, len(trades_today))
            
        except Exception as e:
            logger.error(f"❌ Erreur save_daily_stats: {e}")
            if self.metrics:
                self.metrics.record_error('database', 'save_daily_stats')
    
    def check_performance_alerts(self):
        """Vérifier alertes performance"""
        if not ALERTS_AVAILABLE or not DB_AVAILABLE:
            return
        
        try:
            win_loss = get_win_loss_ratio(days=7)
            if win_loss['win_rate'] < 50 and win_loss['total'] > 10:
                alert_performance_warning(win_loss['win_rate'], 7)
                
                if self.metrics:
                    self.metrics.record_alert('WARNING', 'low_win_rate')
        except Exception as e:
            logger.error(f"❌ Erreur check_performance_alerts: {e}")
    
    def run(self, check_interval_minutes=60):
        """Boucle principale"""
        logger.info("\n" + "="*70)
        logger.info("🚀 LIVE TRADER - VERSION COMPLÈTE")
        logger.info("="*70)
        logger.info(f"⏱️  Intervalle: {check_interval_minutes} min")
        logger.info(f"📊 Mode: {'Paper' if self.paper_trading else '🔴 LIVE'}")
        
        if not self.paper_trading:
            logger.warning("⚠️  MODE LIVE - REAL MONEY!")
            response = input("Continuer? (yes/no): ")
            if response.lower() != 'yes':
                return
        
        if ALERTS_AVAILABLE:
            alert_startup()
        
        cycle = 0
        last_daily_reset = date.today()
        
        try:
            while True:
                cycle += 1
                current_date = date.today()
                
                # Reset quotidien
                if current_date > last_daily_reset and RISK_AVAILABLE:
                    account = self.alpaca.get_account()
                    self.risk_manager.reset_daily_stats(account['portfolio_value'])
                    last_daily_reset = current_date
                    logger.info("🔄 Nouveau jour - Stats réinitialisées")
                
                logger.info(f"\n📍 Cycle {cycle}")
                
                self.check_risk_management()
                self.execute_signals()
                
                if cycle % 4 == 0:
                    self.save_daily_stats()
                
                if cycle % 12 == 0:
                    self.check_performance_alerts()
                
                logger.info(f"\n⏳ Prochain cycle dans {check_interval_minutes} min...")
                time.sleep(check_interval_minutes * 60)
        
        except KeyboardInterrupt:
            logger.info("\n\n🛑 Arrêt manuel")
        except Exception as e:
            logger.error(f"\n❌ Erreur: {e}", exc_info=True)
            
            if self.metrics:
                self.metrics.record_error('system', 'critical_error')
            
            if ALERTS_AVAILABLE:
                send_alert(f"🚨 **ERREUR CRITIQUE**\n\n{str(e)[:200]}", priority='ERROR')
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
            
            if ALERTS_AVAILABLE:
                alert_shutdown(final_value, total_pl)
            
            if DB_AVAILABLE:
                self.save_daily_stats()