"""
Bot de trading autonome pour production
Utilise le modèle universel entraîné
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from datetime import datetime
import json
from stable_baselines3 import PPO
from core.universal_environment import UniversalTradingEnv
from core.market_regime import MarketRegimeDetector
import alpaca_trade_api as tradeapi

class AutonomousTrader:
    """Bot de trading autonome en production"""
    
    def __init__(self):
        # Charger config
        with open('models/autonomous/config_latest.json', 'r') as f:
            self.config = json.load(f)
        
        self.tickers = self.config['assets']
        
        # Charger modèle
        self.model = PPO.load('models/autonomous/production.zip')
        print(f"✅ Modèle chargé : {len(self.tickers)} assets")
        
        # Initialiser Alpaca
        self.api = tradeapi.REST(
            key_id=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url='https://paper-api.alpaca.markets'  # Paper trading
        )
        
        # Market regime detector
        self.regime_detector = MarketRegimeDetector()
        
        print("🤖 Autonomous Trader initialisé\n")
    
    def run(self):
        """Boucle principale"""
        
        print("🚀 Démarrage trading autonome...\n")
        
        while True:
            try:
                # Vérifier si marché ouvert
                clock = self.api.get_clock()
                
                if not clock.is_open:
                    print(f"⏸️ Marché fermé, attente réouverture...")
                    time.sleep(60)
                    continue
                
                # Faire prédiction
                self._trade_cycle()
                
                # Attendre 5 minutes
                time.sleep(300)
                
            except KeyboardInterrupt:
                print("\n⏹️ Arrêt demandé par utilisateur")
                break
            except Exception as e:
                print(f"❌ Erreur : {e}")
                time.sleep(60)
    
    def _trade_cycle(self):
        """Un cycle de trading"""
        
        # Créer environnement
        env = UniversalTradingEnv(
            tickers=self.tickers,
            regime_detector=self.regime_detector
        )
        
        obs, _ = env.reset()
        
        # Prédire actions
        actions, _ = self.model.predict(obs, deterministic=True)
        
        # Exécuter sur Alpaca
        for idx, ticker in enumerate(self.tickers):
            action = actions[idx]
            
            if action == 1:  # BUY
                self._execute_buy(ticker)
            elif action == 2:  # SELL
                self._execute_sell(ticker)
        
        print(f"✅ Cycle {datetime.now().strftime('%H:%M:%S')}")
    
    def _execute_buy(self, ticker):
        """Achète un ticker"""
        try:
            # Vérifier si déjà position
            try:
                position = self.api.get_position(ticker)
                print(f"  ⏭️ {ticker} déjà en position, skip BUY")
                return
            except:
                pass  # Pas de position
            
            # Calculer quantité
            account = self.api.get_account()
            cash = float(account.cash)
            quote = self.api.get_last_trade(ticker)
            price = float(quote.price)
            
            # Diviser cash par nombre d'assets
            max_invest = cash / len(self.tickers)
            qty = int(max_invest / price)
            
            if qty > 0:
                self.api.submit_order(
                    symbol=ticker,
                    qty=qty,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
                print(f"  🟢 BUY {ticker}: {qty} @ ${price:.2f}")
        
        except Exception as e:
            print(f"  ⚠️ BUY {ticker} failed: {e}")
    
    def _execute_sell(self, ticker):
        """Vend un ticker"""
        try:
            # Vérifier position
            try:
                position = self.api.get_position(ticker)
            except:
                print(f"  ⏭️ {ticker} pas de position, skip SELL")
                return
            
            qty = int(position.qty)
            
            if qty > 0:
                self.api.submit_order(
                    symbol=ticker,
                    qty=qty,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                print(f"  🔴 SELL {ticker}: {qty} @ ${position.current_price}")
        
        except Exception as e:
            print(f"  ⚠️ SELL {ticker} failed: {e}")

def main():
    trader = AutonomousTrader()
    trader.run()

if __name__ == "__main__":
    main()
