#!/usr/bin/env python3
"""
🚀 PLOUTOS TRADER V2 - VERSION SIMPLIFIÉE

Bot de trading utilisant le modèle V2 (UniversalTradingEnvV2)
Sans dépendances complexes (BrainTrader, PortfolioManager, etc.)

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ✅ CRÉER DOSSIERS NÉCESSAIRES
os.makedirs('logs', exist_ok=True)
os.makedirs('data_cache', exist_ok=True)

import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

try:
    from core.universal_environment_v2 import UniversalTradingEnvV2
except ImportError:
    print("⚠️  Utilise UniversalTradingEnv V1")
    from core.universal_environment import UniversalTradingEnv as UniversalTradingEnvV2

try:
    from core.alpaca_data_fetcher import AlpacaDataFetcher
    ALPACA_DATA_AVAILABLE = True
except ImportError:
    print("AlpacaDataFetcher non disponible")
    ALPACA_DATA_AVAILABLE = False

try:
    from trading.broker_factory import create_broker
    BROKER_AVAILABLE = True
except ImportError:
    BROKER_AVAILABLE = False

# Compat: garder le flag pour le reste du code
ALPACA_TRADING_AVAILABLE = BROKER_AVAILABLE

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trader_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleTradingBot:
    """
    Bot de trading simplifié utilisant le modèle V2
    """
    
    def __init__(self, model_path='models/autonomous/production.zip', paper_trading=True):
        """
        Args:
            model_path: Chemin vers le modèle PPO
            paper_trading: Mode paper trading (True) ou live (False)
        """
        self.model_path = model_path
        self.paper_trading = paper_trading
        
        # Charger modèle
        logger.info(f"🧠 Chargement modèle: {model_path}")
        try:
            self.model = PPO.load(model_path)
            logger.info("✅ Modèle chargé")
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}")
            raise
        
        # Charger config
        config_path = model_path.replace('.zip', '.json')
        self.tickers = [
            'NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN',
            'SPY', 'QQQ', 'VOO', 'XLE', 'XLF'
        ]
        
        if os.path.exists(config_path):
            import json
            with open(config_path) as f:
                config = json.load(f)
                self.tickers = config.get('tickers', self.tickers)
        
        logger.info(f"🎯 Tickers: {', '.join(self.tickers)}")
        
        # Initialiser Alpaca Data Fetcher
        self.data_fetcher = None
        if ALPACA_DATA_AVAILABLE:
            try:
                self.data_fetcher = AlpacaDataFetcher()
                logger.info("✅ Alpaca Data Fetcher initialisé")
            except Exception as e:
                logger.warning(f"⚠️  Alpaca Data non disponible: {e}")
        
        # Client broker (eToro par défaut, ou Alpaca)
        self.client = None
        if BROKER_AVAILABLE:
            try:
                broker_name = os.getenv('BROKER', 'etoro')
                self.client = create_broker(broker_name, paper_trading=paper_trading)
                logger.info(f"Broker {broker_name} connecte (Paper: {paper_trading})")

                # Afficher infos compte
                account = self.client.get_account()
                if account:
                    logger.info(f"Compte {broker_name}:")
                    logger.info(f"  Cash: ${account['cash']:,.2f}")
                    logger.info(f"  Portfolio: ${account['portfolio_value']:,.2f}")
                    logger.info(f"  Buying Power: ${account['buying_power']:,.2f}")
            except Exception as e:
                logger.warning(f"Broker non disponible: {e}")
                import traceback
                traceback.print_exc()
        
        # Positions actuelles
        self.positions = {}
        self.cash = 100000  # Capital initial (sera mis à jour avec Alpaca)
        
        # Si client connecté, récupérer le vrai solde
        if self.client:
            account = self.client.get_account()
            if account:
                self.cash = account['cash']
    
    def sync_with_broker(self):
        """
        Synchroniser positions et cash avec le broker.
        A appeler AVANT les predictions.
        """
        if not self.client:
            return

        try:
            account = self.client.get_account()
            if account:
                self.cash = float(account['cash'])

            positions = self.client.get_positions()
            self.positions = {pos['symbol']: float(pos['qty']) for pos in positions}

            logger.info(f"Sync broker: ${self.cash:,.2f} cash, {len(self.positions)} positions")

        except Exception as e:
            logger.error(f"Erreur sync broker: {e}")

    # Alias pour compatibilité
    sync_with_alpaca = sync_with_broker
    
    def get_market_data(self, days=30):
        """
        Télécharge données récentes
        Priorité: 1) Cache, 2) Alpaca, 3) yfinance
        
        Args:
            days: Nombre de jours à charger
        
        Returns:
            dict: Données par ticker
        """
        logger.info(f"📡 Chargement données ({days} jours)...")
        
        data = {}
        tickers_to_fetch = []
        
        # ESSAYER CACHE D'ABORD
        for ticker in self.tickers:
            cache_file = f'data_cache/{ticker}.csv'
            
            if os.path.exists(cache_file):
                try:
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    
                    # Vérifier fraîcheur (pas plus de 7 jours)
                    if not df.empty:
                        last_date = pd.to_datetime(df.index[-1])
                        
                        # ✅ FIX: Forcer timezone-naive
                        if last_date.tzinfo:
                            last_date = last_date.tz_localize(None)
                        
                        age_days = (datetime.now() - last_date).days
                        
                        if age_days <= 7 and len(df) >= days:
                            # Garder seulement les derniers jours demandés
                            data[ticker] = df.tail(days)
                            logger.info(f"  ✅ {ticker}: Cache ({len(data[ticker])} jours, age: {age_days}j)")
                            continue
                
                except Exception as e:
                    logger.warning(f"  ⚠️  {ticker}: Erreur lecture cache - {e}")
            
            # Pas en cache ou trop vieux
            tickers_to_fetch.append(ticker)
        
        # TÉLÉCHARGER LES TICKERS MANQUANTS
        if tickers_to_fetch:
            logger.info(f"📡 Téléchargement de {len(tickers_to_fetch)} tickers...")
            
            # PRIORITÉ 1: ALPACA
            if self.data_fetcher:
                try:
                    alpaca_data = self.data_fetcher.fetch_multiple(
                        tickers_to_fetch,
                        days=days,
                        save_cache=True
                    )
                    data.update(alpaca_data)
                    logger.info(f"✅ Alpaca: {len(alpaca_data)} tickers téléchargés")
                    
                    # Mettre à jour liste des manquants
                    tickers_to_fetch = [t for t in tickers_to_fetch if t not in alpaca_data]
                    
                except Exception as e:
                    logger.warning(f"⚠️  Erreur Alpaca: {e}")
            
            # PRIORITÉ 2: YFINANCE (FALLBACK)
            if tickers_to_fetch:
                logger.info(f"📡 Fallback yfinance pour {len(tickers_to_fetch)} tickers...")
                
                for ticker in tickers_to_fetch:
                    df = self._download_with_yfinance(ticker, days)
                    if df is not None and not df.empty:
                        data[ticker] = df
                        # Sauvegarder en cache
                        try:
                            df.to_csv(f'data_cache/{ticker}.csv')
                        except OSError:
                            pass
        
        logger.info(f"✅ {len(data)}/{len(self.tickers)} tickers chargés")
        return data
    
    def _download_with_yfinance(self, ticker, days):
        """
        Télécharge avec yfinance (fallback)
        
        Args:
            ticker: Symbole ticker
            days: Nombre de jours
        
        Returns:
            DataFrame ou None
        """
        try:
            import yfinance as yf
            
            end = datetime.now()
            start = end - timedelta(days=days)
            
            stock = yf.Ticker(ticker)
            df = stock.history(start=start, end=end, interval='1d')
            
            if df.empty or len(df) < 10:
                logger.warning(f"  ⚠️  {ticker}: Données insuffisantes (yfinance)")
                return None
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            logger.info(f"  ✅ {ticker}: yfinance ({len(df)} jours)")
            return df
            
        except Exception as e:
            logger.error(f"  ❌ {ticker}: yfinance - {e}")
            return None
    
    def get_predictions(self, data):
        """
        Génère prédictions avec le modèle
        
        Args:
            data: Données marché
        
        Returns:
            dict: Actions par ticker {ticker: action}
        """
        logger.info("🔮 Génération prédictions...")
        
        try:
            # Sync avec broker avant predictions
            self.sync_with_broker()
            
            # ✅ FIX: Calculer max_steps adapté aux données
            min_data_length = min(len(df) for df in data.values())
            
            # Pour trading live, on n'a besoin que de quelques steps
            max_steps = min(10, max(1, min_data_length - 105))  # Laisse marge pour random start
            
            logger.debug(f"  Data length: {min_data_length}, max_steps: {max_steps}")
            
            # Créer env temporaire
            env = UniversalTradingEnvV2(
                data=data,
                initial_balance=self.cash,
                commission=0.0001,
                max_steps=max_steps,
                buy_pct=0.2
            )
            
            obs, _ = env.reset()
            actions, _ = self.model.predict(obs, deterministic=True)
            
            # Mapper actions
            predictions = {}
            action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            
            for i, ticker in enumerate(env.tickers):
                action = int(actions[i])
                predicted_action = action_map[action]
                
                # ★ FILTRER LES PRÉDICTIONS IMPOSSIBLES
                current_position = self.positions.get(ticker, 0)
                
                if predicted_action == 'SELL' and current_position == 0:
                    logger.debug(f"  ⚠️  {ticker}: SELL prédit mais pas de position → HOLD")
                    predicted_action = 'HOLD'
                
                elif predicted_action == 'BUY' and current_position > 0:
                    logger.debug(f"  ⚠️  {ticker}: BUY prédit mais déjà en position → HOLD")
                    predicted_action = 'HOLD'
                
                predictions[ticker] = predicted_action
            
            # Stats
            stats = {a: sum(1 for v in predictions.values() if v == a) for a in ['BUY', 'SELL', 'HOLD']}
            logger.info(f"✅ Prédictions: {stats['BUY']} BUY | {stats['SELL']} SELL | {stats['HOLD']} HOLD")
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Erreur prédictions: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def execute_trades(self, predictions, current_prices):
        """
        Exécute les trades
        
        Args:
            predictions: Dict {ticker: action}
            current_prices: Dict {ticker: price}
        """
        logger.info("💼 Exécution trades...")
        
        trades_executed = {'buy': 0, 'sell': 0, 'hold': 0}
        
        # ★ VÉRIFIER SYNC (normalement déjà fait dans get_predictions)
        if self.client:
            logger.info(f"  Positions actuelles: {len(self.positions)} actives")
        
        for ticker, action in predictions.items():
            price = current_prices.get(ticker)
            
            if price is None:
                logger.warning(f"⚠️  {ticker}: Prix indisponible")
                continue
            
            try:
                if action == 'BUY':
                    # Acheter si on n'a pas de position
                    current_position = self.positions.get(ticker, 0)
                    
                    if current_position == 0:
                        qty = int((self.cash * 0.1) / price)  # 10% du capital
                        
                        if qty > 0:
                            # ★ TRADE RÉEL AVEC ALPACA
                            if self.client:
                                try:
                                    order = self.client.place_market_order(
                                        symbol=ticker,
                                        qty=qty,
                                        side='buy',
                                        reason='Prédiction modèle IA'
                                    )
                                    
                                    if order:
                                        self.positions[ticker] = qty
                                        self.cash -= qty * price
                                        trades_executed['buy'] += 1
                                        logger.info(f"BUY {ticker}: {qty} @ ${price:.2f} [BROKER]")
                                    else:
                                        logger.error(f"❌ {ticker}: Échec ordre BUY")
                                except Exception as e:
                                    logger.error(f"❌ {ticker}: Erreur BUY - {e}")
                            else:
                                # Simulation seulement
                                self.positions[ticker] = qty
                                self.cash -= qty * price
                                trades_executed['buy'] += 1
                                logger.info(f"✅ {ticker}: BUY {qty} @ ${price:.2f} [SIMULATION]")
                    else:
                        trades_executed['hold'] += 1
                        logger.debug(f"  {ticker}: Déjà en position ({current_position} shares)")
                
                elif action == 'SELL':
                    # Vendre si on a une position
                    current_position = self.positions.get(ticker, 0)
                    
                    if current_position > 0:
                        # ★ TRADE RÉEL AVEC ALPACA
                        if self.client:
                            try:
                                success = self.client.close_position(
                                    symbol=ticker,
                                    reason='Prédiction modèle IA'
                                )
                                
                                if success:
                                    self.cash += current_position * price
                                    self.positions[ticker] = 0
                                    trades_executed['sell'] += 1
                                    logger.info(f"SELL {ticker}: {current_position} @ ${price:.2f} [BROKER]")
                                else:
                                    logger.error(f"❌ {ticker}: Échec SELL")
                            except Exception as e:
                                logger.error(f"❌ {ticker}: Erreur SELL - {e}")
                        else:
                            # Simulation seulement
                            self.cash += current_position * price
                            self.positions[ticker] = 0
                            trades_executed['sell'] += 1
                            logger.info(f"✅ {ticker}: SELL {current_position} @ ${price:.2f} [SIMULATION]")
                    else:
                        trades_executed['hold'] += 1
                        logger.debug(f"  {ticker}: Pas de position à vendre")
                
                else:  # HOLD
                    trades_executed['hold'] += 1
                    
            except Exception as e:
                logger.error(f"❌ {ticker}: Erreur trade - {e}")
        
        logger.info(f"✅ Trades: {trades_executed['buy']} BUY | {trades_executed['sell']} SELL | {trades_executed['hold']} HOLD")
        
        # Portfolio summary
        if self.client:
            # Utiliser valeurs broker reelles
            account = self.client.get_account()
            if account:
                logger.info(f"Portfolio: ${account['portfolio_value']:,.2f} (Cash: ${account['cash']:,.2f})")
                self.cash = account['cash']
        else:
            # Simulation
            total_value = self.cash + sum(
                self.positions.get(t, 0) * current_prices.get(t, 0)
                for t in self.tickers
            )
            logger.info(f"💰 Portfolio Simulation: ${total_value:,.2f} (Cash: ${self.cash:,.2f})")
    
    def run_cycle(self):
        """
        Exécute un cycle de trading complet
        """
        logger.info("\n" + "="*70)
        logger.info(f"🔄 CYCLE DE TRADING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*70)
        
        try:
            # 1. Charger données
            data = self.get_market_data(days=30)
            
            if not data:
                logger.warning("⚠️  Pas de données - cycle annulé")
                return
            
            # 2. Générer prédictions (avec sync Alpaca intégré)
            predictions = self.get_predictions(data)
            
            if not predictions:
                logger.warning("⚠️  Pas de prédictions - cycle annulé")
                return
            
            # 3. Obtenir prix actuels
            current_prices = {ticker: float(df['Close'].iloc[-1]) for ticker, df in data.items()}
            
            # 4. Exécuter trades
            self.execute_trades(predictions, current_prices)
            
            logger.info("="*70 + "\n")
            
        except Exception as e:
            logger.error(f"❌ Erreur cycle: {e}")
            import traceback
            traceback.print_exc()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Ploutos Trading Bot V2 (Simplifié)')
    parser.add_argument('--model', default='models/autonomous/production.zip', help='Chemin modele')
    parser.add_argument('--paper', action='store_true', help='Mode paper trading')
    parser.add_argument('--interval', type=int, default=60, help='Intervalle cycles (minutes)')
    parser.add_argument('--cycles', type=int, default=None, help='Nombre de cycles (illimite par defaut)')
    parser.add_argument('--broker', default=None, choices=['etoro', 'alpaca'],
                        help='Broker a utiliser (defaut: variable BROKER ou etoro)')
    
    args = parser.parse_args()
    
    logger.info("\n" + "="*70)
    logger.info("🚀 PLOUTOS TRADING BOT V2")
    logger.info("="*70)
    broker_name = args.broker or os.getenv('BROKER', 'etoro')
    if args.broker:
        os.environ['BROKER'] = args.broker

    logger.info(f"Modele: {args.model}")
    logger.info(f"Broker: {broker_name}")
    logger.info(f"Mode: {'Paper Trading' if args.paper else 'LIVE TRADING'}")
    logger.info(f"Intervalle: {args.interval} min")
    logger.info("="*70)
    
    if not args.paper:
        logger.warning("⚠️  MODE LIVE - TRADES RÉELS !")
        response = input("Continuer? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("❌ Annulé")
            return
    
    # Initialiser bot
    bot = SimpleTradingBot(model_path=args.model, paper_trading=args.paper)
    
    # Boucle principale
    cycle = 0
    
    try:
        while True:
            cycle += 1
            
            if args.cycles and cycle > args.cycles:
                logger.info(f"\n✅ {args.cycles} cycles terminés")
                break
            
            logger.info(f"\n📍 Cycle {cycle}/{args.cycles or '∞'}")
            
            # Exécuter cycle
            bot.run_cycle()
            
            # Attendre
            if args.cycles is None or cycle < args.cycles:
                logger.info(f"\n⏳ Prochain cycle dans {args.interval} minutes...")
                time.sleep(args.interval * 60)
    
    except KeyboardInterrupt:
        logger.info("\n\n🛑 Arrêt manuel")
    
    except Exception as e:
        logger.error(f"\n❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
