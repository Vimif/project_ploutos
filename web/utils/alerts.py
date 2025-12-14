#!/usr/bin/env python3
"""
🔔 PLOUTOS ALERTS SYSTEM

Système d'alertes temps réel pour conditions de trading
Support: Telegram, Email, Webhooks

Usage:
    alerts = AlertSystem()
    alerts.add_rule('AAPL', condition='rsi_oversold', value=30)
    alerts.check_all()  # Vérifie toutes les règles
"""

import json
import logging
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
import yfinance as yf

try:
    from web.utils.screener import StockScreener
    SCREENER_MODE = True
except:
    SCREENER_MODE = False

logger = logging.getLogger(__name__)


class AlertSystem:
    """
    Système d'alertes multi-canaux
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or 'web/data/alerts_config.json'
        self.rules = self._load_rules()
        self.alert_history = []
        self.screener = StockScreener() if SCREENER_MODE else None
        
        # Configuration Telegram (optionnel)
        self.telegram_bot_token = None
        self.telegram_chat_id = None
        
        logger.info("🔔 Alert System initialisé")
    
    def add_rule(self, ticker: str, condition: str, value: float, 
                 name: Optional[str] = None, channel: str = 'all') -> dict:
        """
        Ajoute une règle d'alerte
        
        Conditions supportées:
            - rsi_oversold: RSI < value
            - rsi_overbought: RSI > value
            - price_above: Prix > value
            - price_below: Prix < value
            - volume_spike: Volume > value * avg
            - pattern_bullish: Pattern haussier détecté
            - pattern_bearish: Pattern baissier détecté
            - breakout_resistance: Cassure résistance
            - breakdown_support: Cassure support
        
        Channels:
            - telegram: Notification Telegram
            - email: Email (si configuré)
            - webhook: Webhook custom
            - all: Tous les canaux actifs
        """
        rule = {
            'id': len(self.rules) + 1,
            'ticker': ticker.upper(),
            'condition': condition,
            'value': value,
            'name': name or f"{ticker} {condition}",
            'channel': channel,
            'active': True,
            'created_at': datetime.now().isoformat(),
            'last_triggered': None,
            'trigger_count': 0
        }
        
        self.rules.append(rule)
        self._save_rules()
        
        logger.info(f"✅ Règle ajoutée: {rule['name']}")
        return rule
    
    def remove_rule(self, rule_id: int) -> bool:
        """
        Supprime une règle
        """
        self.rules = [r for r in self.rules if r['id'] != rule_id]
        self._save_rules()
        return True
    
    def check_all(self) -> List[Dict]:
        """
        Vérifie toutes les règles actives
        Retourne liste des alertes déclenchées
        """
        triggered_alerts = []
        
        for rule in self.rules:
            if not rule['active']:
                continue
            
            try:
                if self._check_rule(rule):
                    alert = self._trigger_alert(rule)
                    triggered_alerts.append(alert)
            except Exception as e:
                logger.error(f"❌ Erreur check rule {rule['id']}: {e}")
        
        return triggered_alerts
    
    def _check_rule(self, rule: dict) -> bool:
        """
        Vérifie si une règle est déclenchée
        """
        ticker = rule['ticker']
        condition = rule['condition']
        value = rule['value']
        
        # Télécharger données récentes
        df = yf.download(ticker, period='5d', progress=False)
        
        if df.empty:
            return False
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        current_price = float(df['Close'].iloc[-1])
        
        # RSI oversold
        if condition == 'rsi_oversold':
            import ta
            rsi = ta.momentum.rsi(df['Close'], window=14).iloc[-1]
            return rsi < value
        
        # RSI overbought
        elif condition == 'rsi_overbought':
            import ta
            rsi = ta.momentum.rsi(df['Close'], window=14).iloc[-1]
            return rsi > value
        
        # Prix au-dessus
        elif condition == 'price_above':
            return current_price > value
        
        # Prix en-dessous
        elif condition == 'price_below':
            return current_price < value
        
        # Volume spike
        elif condition == 'volume_spike':
            current_vol = float(df['Volume'].iloc[-1])
            avg_vol = float(df['Volume'].rolling(20).mean().iloc[-1])
            return current_vol > (avg_vol * value)
        
        # Patterns (nécessite pattern_detector)
        elif condition in ['pattern_bullish', 'pattern_bearish']:
            if self.screener and self.screener.pattern_detector:
                try:
                    patterns = self.screener.pattern_detector.detect_all_patterns(df)
                    recent = patterns.get('candlestick_patterns', [])[-3:]  # 3 derniers
                    
                    for p in recent:
                        if condition == 'pattern_bullish' and p['signal'] == 'BULLISH':
                            return True
                        if condition == 'pattern_bearish' and p['signal'] == 'BEARISH':
                            return True
                except:
                    pass
        
        # Breakout/Breakdown (simplifié)
        elif condition == 'breakout_resistance':
            high_20 = float(df['High'].rolling(20).max().iloc[-2])  # Max des 20 jours précédents
            return current_price > high_20
        
        elif condition == 'breakdown_support':
            low_20 = float(df['Low'].rolling(20).min().iloc[-2])
            return current_price < low_20
        
        return False
    
    def _trigger_alert(self, rule: dict) -> dict:
        """
        Déclenche une alerte
        """
        alert = {
            'rule_id': rule['id'],
            'ticker': rule['ticker'],
            'condition': rule['condition'],
            'name': rule['name'],
            'message': self._generate_message(rule),
            'timestamp': datetime.now().isoformat(),
            'channel': rule['channel']
        }
        
        # Mettre à jour la règle
        rule['last_triggered'] = alert['timestamp']
        rule['trigger_count'] += 1
        self._save_rules()
        
        # Envoyer notification
        self._send_notification(alert)
        
        # Historique
        self.alert_history.append(alert)
        
        logger.info(f"🔔 ALERTE: {alert['message']}")
        
        return alert
    
    def _generate_message(self, rule: dict) -> str:
        """
        Génère le message d'alerte
        """
        ticker = rule['ticker']
        condition = rule['condition']
        value = rule['value']
        
        messages = {
            'rsi_oversold': f"🟢 {ticker}: RSI en SURVENTE (< {value}) - Opportunité d'achat !",
            'rsi_overbought': f"🔴 {ticker}: RSI en SURACHAT (> {value}) - Risque de correction",
            'price_above': f"🚀 {ticker}: Prix a dépassé {value}$ - Breakout confirmé !",
            'price_below': f"📉 {ticker}: Prix est passé sous {value}$ - Breakdown",
            'volume_spike': f"🔥 {ticker}: Volume explosif ({value}x la moyenne) - Mouvement majeur",
            'pattern_bullish': f"🎯 {ticker}: Pattern HAUSSIER détecté - Signal d'achat",
            'pattern_bearish': f"⚠️ {ticker}: Pattern BAISSIER détecté - Signal de vente",
            'breakout_resistance': f"🚀 {ticker}: CASSURE de résistance - Momentum haussier",
            'breakdown_support': f"🚨 {ticker}: CASSURE de support - Momentum baissier"
        }
        
        return messages.get(condition, f"{ticker}: Alerte {condition}")
    
    def _send_notification(self, alert: dict):
        """
        Envoie notification sur les canaux configurés
        """
        channel = alert['channel']
        message = alert['message']
        
        # Telegram
        if channel in ['telegram', 'all'] and self.telegram_bot_token:
            self._send_telegram(message)
        
        # Email (TODO)
        # if channel in ['email', 'all']:
        #     self._send_email(message)
        
        # Webhook (TODO)
        # if channel in ['webhook', 'all']:
        #     self._send_webhook(alert)
    
    def _send_telegram(self, message: str) -> bool:
        """
        Envoie message Telegram
        """
        if not self.telegram_bot_token or not self.telegram_chat_id:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=data, timeout=5)
            
            if response.status_code == 200:
                logger.info("✅ Message Telegram envoyé")
                return True
            else:
                logger.error(f"❌ Telegram error: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur Telegram: {e}")
            return False
    
    def configure_telegram(self, bot_token: str, chat_id: str):
        """
        Configure Telegram
        """
        self.telegram_bot_token = bot_token
        self.telegram_chat_id = chat_id
        logger.info("✅ Telegram configuré")
    
    def get_rules(self) -> List[Dict]:
        """Retourne toutes les règles"""
        return self.rules
    
    def get_history(self, limit: int = 50) -> List[Dict]:
        """Retourne historique des alertes"""
        return self.alert_history[-limit:]
    
    def _load_rules(self) -> List[Dict]:
        """Charge règles depuis fichier"""
        try:
            path = Path(self.config_path)
            if path.exists():
                with open(path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Erreur chargement rules: {e}")
        
        return []
    
    def _save_rules(self):
        """Sauvegarde règles"""
        try:
            path = Path(self.config_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(self.rules, f, indent=2)
        except Exception as e:
            logger.error(f"Erreur sauvegarde rules: {e}")


if __name__ == '__main__':
    # Test
    alerts = AlertSystem()
    
    # Ajouter quelques règles de test
    alerts.add_rule('AAPL', 'rsi_oversold', 30)
    alerts.add_rule('NVDA', 'price_above', 500)
    alerts.add_rule('TSLA', 'volume_spike', 2.0)
    
    print("✅ Règles ajoutées:")
    for rule in alerts.get_rules():
        print(f"  - {rule['name']}")
    
    print("\n🔍 Vérification des règles...")
    triggered = alerts.check_all()
    
    if triggered:
        print(f"\n🔔 {len(triggered)} alerte(s) déclenchée(s):")
        for alert in triggered:
            print(f"  {alert['message']}")
    else:
        print("\n✅ Aucune alerte")
