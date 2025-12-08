# notifications/discord_notifier.py
"""
🔔 DISCORD NOTIFIER

Envoie notifications Discord pour suivi temps réel du bot

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import os
import requests
from datetime import datetime
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

class DiscordNotifier:
    """
    Gestionnaire de notifications Discord
    """
    
    def __init__(self, webhook_url: Optional[str] = None):
        """
        Args:
            webhook_url: URL du webhook Discord (ou depuis .env)
        """
        self.webhook_url = webhook_url or os.getenv('DISCORD_WEBHOOK_URL')
        self.enabled = bool(self.webhook_url)
        
        if self.enabled:
            logger.info("✅ Discord notifications activées")
        else:
            logger.warning("⚠️  Discord notifications désactivées (webhook manquant)")
    
    def send_message(self, content: str = None, embed: Dict = None) -> bool:
        """
        Envoyer message Discord
        
        Args:
            content: Texte simple
            embed: Embed riche (dict)
        
        Returns:
            bool: Succès
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
                timeout=5
            )
            
            if response.status_code == 204:
                logger.debug("✅ Message Discord envoyé")
                return True
            else:
                logger.error(f"❌ Discord erreur {response.status_code}: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur Discord: {e}")
            return False
    
    def notify_cycle_start(self, cycle_number: int):
        """
        Notifier début de cycle
        """
        embed = {
            'title': f"🔄 Cycle #{cycle_number}",
            'description': "Démarrage du cycle de trading",
            'color': 3447003,  # Bleu
            'timestamp': datetime.utcnow().isoformat(),
            'footer': {'text': 'Ploutos Trading Bot'}
        }
        
        return self.send_message(embed=embed)
    
    def notify_trade(self, symbol: str, action: str, quantity: float, 
                     price: float, reason: str = '', success: bool = True):
        """
        Notifier exécution trade
        
        Args:
            symbol: Ticker
            action: 'BUY' ou 'SELL'
            quantity: Nombre d'actions
            price: Prix
            reason: Raison du trade
            success: Trade réussi ou non
        """
        if action == 'BUY':
            emoji = '🟢'
            color = 3066993  # Vert
        else:
            emoji = '🔴'
            color = 15158332  # Rouge
        
        if not success:
            emoji = '❌'
            color = 10038562  # Gris
        
        amount = quantity * price
        
        embed = {
            'title': f"{emoji} {action} {symbol}",
            'description': reason or 'Trade exécuté',
            'color': color,
            'fields': [
                {'name': 'Quantité', 'value': f"{quantity:.2f}", 'inline': True},
                {'name': 'Prix', 'value': f"${price:.2f}", 'inline': True},
                {'name': 'Montant', 'value': f"${amount:,.2f}", 'inline': True}
            ],
            'timestamp': datetime.utcnow().isoformat(),
            'footer': {'text': 'Ploutos Trading Bot'}
        }
        
        return self.send_message(embed=embed)
    
    def notify_cycle_summary(self, trades_count: Dict[str, int], 
                            portfolio_value: float, cash: float,
                            daily_pl: float = None):
        """
        Notifier résumé de cycle
        
        Args:
            trades_count: {'buy': X, 'sell': Y, 'hold': Z}
            portfolio_value: Valeur totale
            cash: Cash disponible
            daily_pl: P&L journalier (optionnel)
        """
        buy_count = trades_count.get('buy', 0)
        sell_count = trades_count.get('sell', 0)
        hold_count = trades_count.get('hold', 0)
        
        fields = [
            {'name': '🟢 BUY', 'value': str(buy_count), 'inline': True},
            {'name': '🔴 SELL', 'value': str(sell_count), 'inline': True},
            {'name': '⏸️ HOLD', 'value': str(hold_count), 'inline': True},
            {'name': '💰 Portfolio', 'value': f"${portfolio_value:,.2f}", 'inline': True},
            {'name': '💵 Cash', 'value': f"${cash:,.2f}", 'inline': True}
        ]
        
        if daily_pl is not None:
            pl_emoji = '📈' if daily_pl >= 0 else '📉'
            fields.append({
                'name': f'{pl_emoji} P&L Jour',
                'value': f"${daily_pl:+,.2f}",
                'inline': True
            })
        
        color = 3066993 if (buy_count + sell_count) > 0 else 10070709  # Vert si trades, gris sinon
        
        embed = {
            'title': '📊 Résumé du Cycle',
            'color': color,
            'fields': fields,
            'timestamp': datetime.utcnow().isoformat(),
            'footer': {'text': 'Ploutos Trading Bot'}
        }
        
        return self.send_message(embed=embed)
    
    def notify_market_closed(self, next_open: str):
        """
        Notifier que le marché est fermé
        
        Args:
            next_open: Heure prochaine ouverture
        """
        embed = {
            'title': '🚪 Marché Fermé',
            'description': f"Le marché est actuellement fermé.\n\nProchaine ouverture: **{next_open}**",
            'color': 10070709,  # Gris
            'timestamp': datetime.utcnow().isoformat(),
            'footer': {'text': 'Ploutos Trading Bot'}
        }
        
        return self.send_message(embed=embed)
    
    def notify_error(self, error_message: str, details: str = ''):
        """
        Notifier erreur critique
        
        Args:
            error_message: Message d'erreur
            details: Détails additionnels
        """
        embed = {
            'title': '❌ Erreur Critique',
            'description': error_message,
            'color': 15158332,  # Rouge
            'timestamp': datetime.utcnow().isoformat(),
            'footer': {'text': 'Ploutos Trading Bot'}
        }
        
        if details:
            embed['fields'] = [{
                'name': 'Détails',
                'value': details[:1024]  # Limite Discord
            }]
        
        return self.send_message(embed=embed)
    
    def notify_with_explanation(self, symbol: str, action: str, 
                               quantity: float, price: float,
                               explanation: Dict):
        """
        Notifier trade avec explication IA
        
        Args:
            symbol: Ticker
            action: 'BUY' ou 'SELL'
            quantity: Quantité
            price: Prix
            explanation: Dict avec 'reason', 'confidence', 'indicators'
        """
        if action == 'BUY':
            emoji = '🟢'
            color = 3066993
        else:
            emoji = '🔴'
            color = 15158332
        
        amount = quantity * price
        reason = explanation.get('reason', 'Aucune raison')
        confidence = explanation.get('confidence', 0)
        indicators = explanation.get('indicators', {})
        
        fields = [
            {'name': 'Quantité', 'value': f"{quantity:.2f}", 'inline': True},
            {'name': 'Prix', 'value': f"${price:.2f}", 'inline': True},
            {'name': 'Montant', 'value': f"${amount:,.2f}", 'inline': True},
            {'name': '🧠 Raison', 'value': reason, 'inline': False},
            {'name': '🎯 Confiance', 'value': f"{confidence:.0f}%", 'inline': True}
        ]
        
        # Ajouter indicateurs
        if indicators:
            indicators_text = '\n'.join([
                f"**{k}**: {v}"
                for k, v in list(indicators.items())[:3]  # Max 3 indicateurs
            ])
            fields.append({
                'name': '📊 Indicateurs',
                'value': indicators_text,
                'inline': False
            })
        
        embed = {
            'title': f"{emoji} {action} {symbol}",
            'color': color,
            'fields': fields,
            'timestamp': datetime.utcnow().isoformat(),
            'footer': {'text': 'Ploutos Trading Bot'}
        }
        
        return self.send_message(embed=embed)

# Helper pour tests
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from dotenv import load_dotenv
    load_dotenv()
    
    notifier = DiscordNotifier()
    
    if notifier.enabled:
        print("📨 Test notifications Discord...\n")
        
        # Test 1: Trade
        notifier.notify_trade('NVDA', 'BUY', 15, 142.50, 'Signal haussier fort')
        
        # Test 2: Résumé
        notifier.notify_cycle_summary(
            {'buy': 2, 'sell': 1, 'hold': 7},
            102435.50,
            -3737.47,
            daily_pl=1245.67
        )
        
        # Test 3: Trade avec explication
        notifier.notify_with_explanation(
            'AAPL', 'SELL', 36, 278.81,
            {
                'reason': 'Prise de profit (+3.2%)',
                'confidence': 85,
                'indicators': {
                    'RSI': '72 (surachat)',
                    'MACD': 'Signal baissier',
                    'Volume': '+15%'
                }
            }
        )
        
        print("✅ Tests envoyés ! Vérifie ton Discord.")
    else:
        print("❌ DISCORD_WEBHOOK_URL non configuré dans .env")
