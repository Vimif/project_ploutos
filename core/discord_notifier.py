#!/usr/bin/env python3
"""
👍 DISCORD NOTIFIER

Envoie des notifications Discord pour le bot de trading

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import requests
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DiscordNotifier:
    """
    Gestionnaire de notifications Discord
    """
    
    def __init__(self, webhook_url=None):
        """
        Args:
            webhook_url: URL du webhook Discord
        """
        self.webhook_url = webhook_url
        self.enabled = bool(webhook_url)
        
        if self.enabled:
            logger.info("✅ Discord notifier activé")
        else:
            logger.warning("⚠️  Discord notifier désactivé (pas de webhook)")
    
    def send_message(self, content=None, embed=None):
        """
        Envoyer un message Discord
        
        Args:
            content: Texte simple
            embed: Embed riche (dict)
        
        Returns:
            bool: Succès/échec
        """
        if not self.enabled:
            return False
        
        try:
            payload = {}
            
            if content:
                payload['content'] = content
            
            if embed:
                payload['embeds'] = [embed]
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 204:
                logger.debug("✅ Message Discord envoyé")
                return True
            else:
                logger.error(f"❌ Discord error: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur envoi Discord: {e}")
            return False
    
    def notify_trade(self, ticker, action, quantity, price, reason='', pnl=None):
        """
        Notifier un trade exécuté
        
        Args:
            ticker: Symbole
            action: 'BUY' ou 'SELL'
            quantity: Quantité
            price: Prix unitaire
            reason: Raison du trade
            pnl: P&L réalisé (pour SELL)
        """
        if not self.enabled:
            return
        
        # Emoji selon action
        emoji = "🟢" if action == 'BUY' else "🔴"
        color = 0x00ff00 if action == 'BUY' else 0xff0000
        
        # Montant total
        amount = quantity * price
        
        # Construire description
        description = f"**{quantity} actions @ ${price:.2f}**\n"
        description += f"Montant total: ${amount:,.2f}\n"
        
        if reason:
            description += f"\n🧠 **Raison:**\n{reason}"
        
        if pnl is not None:
            pnl_emoji = "📈" if pnl >= 0 else "📉"
            description += f"\n\n{pnl_emoji} **P&L:** ${pnl:+,.2f}"
        
        embed = {
            'title': f"{emoji} {action} {ticker}",
            'description': description,
            'color': color,
            'timestamp': datetime.utcnow().isoformat(),
            'footer': {
                'text': 'Ploutos Trading Bot',
                'icon_url': 'https://em-content.zobj.net/thumbs/120/apple/325/robot_1f916.png'
            }
        }
        
        self.send_message(embed=embed)
    
    def notify_cycle_start(self, cycle_num, market_open=True):
        """
        Notifier le début d'un cycle
        
        Args:
            cycle_num: Numéro du cycle
            market_open: Marché ouvert ou fermé
        """
        if not self.enabled:
            return
        
        status_emoji = "🟢" if market_open else "🔴"
        status_text = "Ouvert" if market_open else "Fermé"
        
        embed = {
            'title': f"🔄 Cycle #{cycle_num}",
            'description': f"{status_emoji} Marché: **{status_text}**",
            'color': 0x3498db,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.send_message(embed=embed)
    
    def notify_cycle_summary(self, cycle_num, trades_executed, portfolio_value, cash, positions_count, pnl_daily=None):
        """
        Notifier le résumé d'un cycle
        
        Args:
            cycle_num: Numéro du cycle
            trades_executed: Dict {'buy': x, 'sell': y, 'hold': z}
            portfolio_value: Valeur totale du portfolio
            cash: Cash disponible
            positions_count: Nombre de positions ouvertes
            pnl_daily: P&L journalier
        """
        if not self.enabled:
            return
        
        # Construire description
        description = f"💼 **Trades exécutés:**\n"
        description += f"• 🟢 BUY: {trades_executed['buy']}\n"
        description += f"• 🔴 SELL: {trades_executed['sell']}\n"
        description += f"• ⏸️ HOLD: {trades_executed['hold']}\n"
        
        description += f"\n💰 **Portfolio:**\n"
        description += f"• Valeur totale: ${portfolio_value:,.2f}\n"
        description += f"• Cash: ${cash:,.2f}\n"
        description += f"• Positions: {positions_count}\n"
        
        if pnl_daily is not None:
            pnl_emoji = "📈" if pnl_daily >= 0 else "📉"
            description += f"\n{pnl_emoji} **P&L journalier:** ${pnl_daily:+,.2f}\n"
        
        embed = {
            'title': f"✅ Cycle #{cycle_num} terminé",
            'description': description,
            'color': 0x2ecc71,
            'timestamp': datetime.utcnow().isoformat(),
            'footer': {
                'text': 'Ploutos Trading Bot'
            }
        }
        
        self.send_message(embed=embed)
    
    def notify_error(self, error_type, error_message):
        """
        Notifier une erreur
        
        Args:
            error_type: Type d'erreur
            error_message: Message d'erreur
        """
        if not self.enabled:
            return
        
        embed = {
            'title': f"❌ Erreur: {error_type}",
            'description': f"```{error_message}```",
            'color': 0xe74c3c,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.send_message(embed=embed)
    
    def notify_market_status(self, is_open, next_open=None, next_close=None):
        """
        Notifier le statut du marché
        
        Args:
            is_open: Marché ouvert (bool)
            next_open: Prochaine ouverture (datetime)
            next_close: Prochaine fermeture (datetime)
        """
        if not self.enabled:
            return
        
        status_emoji = "🟢" if is_open else "🔴"
        status_text = "**OUVERT**" if is_open else "**FERMÉ**"
        
        description = f"{status_emoji} Marché US: {status_text}\n\n"
        
        if next_open:
            description += f"🔔 Prochaine ouverture: {next_open.strftime('%Y-%m-%d %H:%M %Z')}\n"
        
        if next_close:
            description += f"🔕 Prochaine fermeture: {next_close.strftime('%Y-%m-%d %H:%M %Z')}\n"
        
        embed = {
            'title': '🏛️ Statut Marché',
            'description': description,
            'color': 0x3498db if is_open else 0x95a5a6,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.send_message(embed=embed)
    
    def notify_prediction_analysis(self, ticker, action, confidence, indicators):
        """
        Notifier l'analyse d'une prédiction avec explications
        
        Args:
            ticker: Symbole
            action: Action prédite
            confidence: Confiance du modèle (0-1)
            indicators: Dict d'indicateurs techniques
        """
        if not self.enabled:
            return
        
        emoji = "🟢" if action == 'BUY' else ("🔴" if action == 'SELL' else "⏸️")
        
        description = f"**Action:** {action}\n"
        description += f"**Confiance:** {confidence*100:.1f}%\n\n"
        
        if indicators:
            description += "📊 **Indicateurs:**\n"
            for key, value in indicators.items():
                description += f"• {key}: {value}\n"
        
        embed = {
            'title': f"{emoji} Analyse {ticker}",
            'description': description,
            'color': 0x9b59b6,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.send_message(embed=embed)

# Fonction helper
def create_discord_notifier(webhook_url=None):
    """
    Créer un notifier Discord
    
    Args:
        webhook_url: URL webhook (ou depuis env DISCORD_WEBHOOK_URL)
    
    Returns:
        DiscordNotifier
    """
    import os
    
    if not webhook_url:
        webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    
    return DiscordNotifier(webhook_url=webhook_url)
