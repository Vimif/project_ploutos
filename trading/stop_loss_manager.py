"""Gestion des stop loss et take profit"""

import logging

logger = logging.getLogger(__name__)

class StopLossManager:
    """Gère stop loss et take profit"""

    def __init__(self, broker_client, stop_loss_pct=0.02, take_profit_pct=0.15):
        self.broker = broker_client
        # Alias pour compatibilité
        self.alpaca = broker_client
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def check_all_positions(self, positions, metrics=None):
        """Vérifie stop loss / take profit sur toutes les positions"""

        for pos in positions:
            symbol = pos['symbol']
            unrealized_plpc = pos['unrealized_plpc']
            unrealized_pl = pos['unrealized_pl']

            if unrealized_plpc <= -self.stop_loss_pct:
                self._execute_stop_loss(symbol, unrealized_pl, unrealized_plpc, metrics)

            elif unrealized_plpc >= self.take_profit_pct:
                self._execute_take_profit(symbol, unrealized_pl, unrealized_plpc, metrics)

    def _execute_stop_loss(self, symbol, pl, pl_pct, metrics):
        """Exécute un stop loss"""
        logger.warning(f"🛑 STOP LOSS: {symbol} ({pl_pct*100:.2f}%)")

        if self.broker.close_position(symbol, reason=f'Stop Loss {pl_pct*100:.1f}%'):
            if metrics:
                metrics.record_trade(symbol, 'SELL', abs(pl), result='loss')
            logger.info(f"✅ {symbol} fermé (Stop Loss)")

    def _execute_take_profit(self, symbol, pl, pl_pct, metrics):
        """Exécute un take profit"""
        logger.info(f"🎯 TAKE PROFIT: {symbol} ({pl_pct*100:.2f}%)")

        if self.broker.close_position(symbol, reason=f'Take Profit {pl_pct*100:.1f}%'):
            if metrics:
                metrics.record_trade(symbol, 'SELL', pl, result='win')
            logger.info(f"✅ {symbol} fermé (Take Profit)")
