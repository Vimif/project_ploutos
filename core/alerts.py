# core/alerts.py
"""Système d'alertes multi-canaux (Telegram + Discord)"""

import sys
from pathlib import Path
if str(Path(__file__).parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import requests
from datetime import datetime
from dotenv import load_dotenv
from core.utils import setup_logging

load_dotenv()
logger = setup_logging(__name__)

class AlertManager:
    """Gestionnaire d'alertes multi-canaux"""
    
    def __init__(self):
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.discord_webhook = os.getenv('DISCORD_WEBHOOK_URL')
        self.alerts_enabled = os.getenv('ALERTS_ENABLED', 'true').lower() == 'true'
        
        if self.alerts_enabled:
            if self.telegram_token and self.telegram_chat_id:
                logger.info("✅ Alertes Telegram activées")
            if self.discord_webhook:
                logger.info("✅ Alertes Discord activées")
        else:
            logger.info("⚠️  Alertes désactivées")
    
    def send_telegram(self, message: str, priority: str = 'INFO'):
        """Envoyer alerte Telegram"""
        if not self.alerts_enabled:
            return
        
        if not self.telegram_token or not self.telegram_chat_id:
            return
        
        emoji_map = {
            'INFO': 'ℹ️',
            'SUCCESS': '✅',
            'WARNING': '⚠️',
            'ERROR': '🚨',
            'TRADE': '💰',
            'PROFIT': '📈',
            'LOSS': '📉'
        }
        
        emoji = emoji_map.get(priority, 'ℹ️')
        formatted = f"{emoji} *PLOUTOS TRADING*\n\n{message}\n\n_({datetime.now().strftime('%H:%M:%S')})_"
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            response = requests.post(url, json={
                'chat_id': self.telegram_chat_id,
                'text': formatted,
                'parse_mode': 'Markdown'
            }, timeout=5)
            
            if response.status_code == 200:
                logger.debug(f"✅ Alerte Telegram envoyée: {priority}")
            else:
                logger.error(f"❌ Erreur Telegram: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Erreur Telegram: {e}")
    
    def send_discord(self, message: str, priority: str = 'INFO'):
        """Envoyer alerte Discord"""
        if not self.alerts_enabled:
            return
        
        if not self.discord_webhook:
            return
        
        color_map = {
            'INFO': 3447003,      # Bleu
            'SUCCESS': 3066993,   # Vert
            'WARNING': 15105570,  # Orange
            'ERROR': 15158332,    # Rouge
            'TRADE': 10181046,    # Violet
            'PROFIT': 3066993,    # Vert
            'LOSS': 15158332      # Rouge
        }
        
        try:
            requests.post(self.discord_webhook, json={
                'embeds': [{
                    'title': f'🤖 Ploutos Trading - {priority}',
                    'description': message,
                    'color': color_map.get(priority, 3447003),
                    'timestamp': datetime.utcnow().isoformat(),
                    'footer': {'text': 'Ploutos Trading Bot'}
                }]
            }, timeout=5)
            
            logger.debug(f"✅ Alerte Discord envoyée: {priority}")
            
        except Exception as e:
            logger.error(f"❌ Erreur Discord: {e}")
    
    def alert(self, message: str, priority: str = 'INFO', channels: list = None):
        """
        Envoyer alerte sur les canaux spécifiés
        
        Args:
            message: Message à envoyer
            priority: INFO, SUCCESS, WARNING, ERROR, TRADE, PROFIT, LOSS
            channels: ['telegram', 'discord'] ou None pour tous
        """
        if not self.alerts_enabled:
            return
        
        if channels is None:
            channels = ['telegram', 'discord']
        
        if 'telegram' in channels:
            self.send_telegram(message, priority)
        
        if 'discord' in channels:
            self.send_discord(message, priority)

# Instance globale
_alert_manager = None

def get_alert_manager():
    """Obtenir instance AlertManager (singleton)"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager

def send_alert(message: str, priority: str = 'INFO', channels: list = None):
    """Helper pour envoyer une alerte rapidement"""
    manager = get_alert_manager()
    manager.alert(message, priority, channels)

# Helpers spécifiques
def alert_trade(symbol: str, action: str, quantity: float, price: float, amount: float):
    """Alerte pour un trade"""
    send_alert(
        f"**{action}** {quantity} {symbol}\n"
        f"Prix: ${price:.2f}\n"
        f"Montant: ${amount:,.2f}",
        priority='TRADE'
    )

def alert_profit(symbol: str, pl: float, pl_pct: float):
    """Alerte pour un profit"""
    send_alert(
        f"💰 **PROFIT sur {symbol}**\n"
        f"P&L: ${pl:+,.2f} ({pl_pct:+.2f}%)",
        priority='PROFIT'
    )

def alert_loss(symbol: str, pl: float, pl_pct: float):
    """Alerte pour une perte"""
    send_alert(
        f"📉 **PERTE sur {symbol}**\n"
        f"P&L: ${pl:+,.2f} ({pl_pct:+.2f}%)",
        priority='LOSS'
    )

def alert_daily_summary(portfolio_value: float, pl: float, pl_pct: float, trades_count: int):
    """Alerte résumé quotidien"""
    priority = 'PROFIT' if pl > 0 else 'LOSS' if pl < 0 else 'INFO'
    send_alert(
        f"📊 **RÉSUMÉ QUOTIDIEN**\n\n"
        f"Portfolio: ${portfolio_value:,.2f}\n"
        f"P&L: ${pl:+,.2f} ({pl_pct:+.2f}%)\n"
        f"Trades: {trades_count}",
        priority=priority
    )

def alert_performance_warning(win_rate: float, days: int):
    """Alerte performance faible"""
    send_alert(
        f"⚠️ **WIN RATE FAIBLE**\n\n"
        f"Win rate {days}j: {win_rate:.1f}%\n"
        f"Révision stratégie recommandée",
        priority='WARNING'
    )

def alert_startup():
    """Alerte démarrage du bot"""
    send_alert(
        f"🚀 **BOT DÉMARRÉ**\n\n"
        f"Mode: Paper Trading\n"
        f"Statut: Opérationnel",
        priority='SUCCESS'
    )

def alert_shutdown(portfolio_value: float, total_pl: float):
    """Alerte arrêt du bot"""
    send_alert(
        f"🛑 **BOT ARRÊTÉ**\n\n"
        f"Portfolio final: ${portfolio_value:,.2f}\n"
        f"P&L total: ${total_pl:+,.2f}",
        priority='INFO'
    )