#!/bin/bash
# 🔧 PATCH: Intégrer Market Status + Discord Notifier

echo "🔧 Application patch intégrations..."

# Backup
cp scripts/run_trader_v2_simple.py scripts/run_trader_v2_simple.py.backup
echo "✅ Backup créé"

# Patch le bot
python3 << 'PYTHON_EOF'
with open('scripts/run_trader_v2_simple.py', 'r') as f:
    content = f.read()

# 1. Ajouter imports après les imports existants
import_section = """
try:
    from core.market_status import MarketStatus
    MARKET_STATUS_AVAILABLE = True
except ImportError:
    MARKET_STATUS_AVAILABLE = False
    print("⚠️  MarketStatus non disponible")

try:
    from notifications.discord_notifier import DiscordNotifier
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    print("⚠️  Discord Notifier non disponible")
"""

# Insérer après "logger = logging.getLogger(__name__)"
if "MARKET_STATUS_AVAILABLE" not in content:
    content = content.replace(
        "logger = logging.getLogger(__name__)",
        "logger = logging.getLogger(__name__)\n" + import_section
    )

# 2. Ajouter dans __init__ après self.client
init_addition = """
        
        # Discord Notifier
        self.discord = None
        if DISCORD_AVAILABLE:
            try:
                self.discord = DiscordNotifier()
            except Exception as e:
                logger.warning(f"⚠️  Discord non disponible: {e}")
        
        # Market Status Checker
        self.market_checker = None
        if MARKET_STATUS_AVAILABLE and self.client:
            self.market_checker = MarketStatus(self.client)
"""

# Insérer après "self.cash = account['cash']" dans __init__
if "self.discord" not in content:
    content = content.replace(
        "                self.cash = account['cash']",
        "                self.cash = account['cash']" + init_addition
    )

# 3. Ajouter vérification marché dans run_cycle
market_check = """
            # ★ VÉRIFIER SI MARCHÉ OUVERT
            if self.market_checker:
                market_status = self.market_checker.get_market_status(use_cache=True)
                
                if not market_status.get('is_open', False):
                    time_until = market_status.get('time_until_open', '?')
                    logger.warning(f"⏸️  Marché fermé. Ouverture dans: {time_until}")
                    
                    if self.discord:
                        self.discord.notify_market_closed(time_until)
                    
                    return
            
"""

# Insérer au début de run_cycle après le logger.info
if "VÉRIFIER SI MARCHÉ OUVERT" not in content:
    content = content.replace(
        '        logger.info("="*70)\n        \n        try:',
        '        logger.info("="*70)\n        \n        try:' + market_check
    )

# 4. Ajouter notification Discord après execute_trades
discord_notify = """
            
            # ★ NOTIFIER DISCORD
            if self.discord:
                trades_list = []
                
                for ticker, action in predictions.items():
                    if action in ['BUY', 'SELL']:
                        trades_list.append({
                            'ticker': ticker,
                            'action': action,
                            'qty': int((self.cash * 0.1) / current_prices[ticker]) if action == 'BUY' else self.positions.get(ticker, 0),
                            'price': current_prices[ticker]
                        })
                
                if trades_list:
                    self.discord.notify_trades(
                        trades_list,
                        data,
                        self.portfolio_value,
                        self.cash
                    )
"""

# Insérer après execute_trades
if "NOTIFIER DISCORD" not in content:
    content = content.replace(
        '            self.execute_trades(predictions, current_prices)\n            \n            logger.info("="*70',
        '            self.execute_trades(predictions, current_prices)' + discord_notify + '\n            \n            logger.info("="*70'
    )

# Sauvegarder
with open('scripts/run_trader_v2_simple.py', 'w') as f:
    f.write(content)

print("✅ Patch appliqué")

PYTHON_EOF

echo "✅ Bot patché avec succès"
echo ""
echo "📋 Changements:"
echo "  - ✅ Import Market Status + Discord"
echo "  - ✅ Initialisation dans __init__"
echo "  - ✅ Vérification marché ouvert"
echo "  - ✅ Notifications Discord trades"
echo ""
echo "🔄 Redémarre le bot pour appliquer:"
echo "   sudo systemctl restart ploutos-trader-v2"

