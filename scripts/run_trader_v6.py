#!/usr/bin/env python3
"""
🚀 PLOUTOS TRADER V6 - BETTER TIMING
Bot de trading utilisant le modèle V6 (UniversalTradingEnvV6BetterTiming)
Optimisé pour éviter le "buying high" avec les features V2.

Auteur: Ploutos AI Team
Date: Feb 2026
"""

import sys
import os
from pathlib import Path

# Fix encoding for Windows console (emojis)
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Ajouter la racine du projet au path
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

# ✅ Import Environnement V6
try:
    from core.universal_environment_v6_better_timing import UniversalTradingEnvV6BetterTiming
except ImportError as e:
    print(f"❌ Erreur import UniversalTradingEnvV6BetterTiming: {e}")
    sys.exit(1)

# Import Data Fetcher
try:
    from core.alpaca_data_fetcher import AlpacaDataFetcher
    ALPACA_DATA_AVAILABLE = True
except ImportError:
    print("⚠️ AlpacaDataFetcher non disponible")
    ALPACA_DATA_AVAILABLE = False

# Import Broker Factory
try:
    from trading.broker_factory import create_broker
    BROKER_AVAILABLE = True
except ImportError:
    BROKER_AVAILABLE = False

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trader_v6.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TraderV6")

class TraderV6:
    """
    Bot de trading V6 utilisant le modèle 'Better Timing'
    """
    
    def __init__(self, model_path='data/models/brain_tech.zip', paper_trading=True, tickers=None):
        """
        Args:
            model_path: Chemin vers le modèle PPO
            paper_trading: Mode paper trading (True) ou live (False)
            tickers: Liste optionnelle de tickers (sinon charge depuis config ou défaut)
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
        
        # Définir tickers
        if tickers:
            self.tickers = tickers
        else:
            # Essayer de charger config json associée au modèle
            config_path = model_path.replace('.zip', '.json')
            if os.path.exists(config_path):
                import json
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                        # Chercher tickers dans différentes structures possibles
                        if 'tickers' in config:
                            self.tickers = config['tickers']
                        elif 'data' in config and 'tickers' in config['data']:
                            self.tickers = config['data']['tickers']
                        else:
                            self.tickers = ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA', 'AMD', 'INTC']
                except Exception:
                    self.tickers = ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA', 'AMD', 'INTC']
            else:
                # Défaut Tech
                self.tickers = ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA', 'AMD', 'INTC']
        
        logger.info(f"🎯 Tickers cibles: {', '.join(self.tickers)}")
        
        # Initialiser Data Fetcher
        self.data_fetcher = None
        if ALPACA_DATA_AVAILABLE:
            try:
                self.data_fetcher = AlpacaDataFetcher()
                logger.info("✅ Alpaca Data Fetcher initialisé")
            except Exception as e:
                logger.warning(f"⚠️ Alpaca Data non disponible: {e}")
        
        # Initialiser Broker Client
        self.client = None
        if BROKER_AVAILABLE:
            try:
                broker_name = os.getenv('BROKER', 'etoro')
                self.client = create_broker(broker_name, paper_trading=paper_trading)
                logger.info(f"🔌 Broker {broker_name} connecté (Paper: {paper_trading})")
                
                # Afficher infos compte
                account = self.client.get_account()
                if account:
                    logger.info(f"💰 Compte {broker_name}:")
                    logger.info(f"   • Cash: ${account.get('cash', 0):,.2f}")
                    logger.info(f"   • Portfolio: ${account.get('portfolio_value', 0):,.2f}")
            except Exception as e:
                logger.warning(f"⚠️ Broker non disponible ou erreur connexion: {e}")
        
        # État interne
        self.positions = {}
        self.cash = 100000.0
        
        # Sync initiale
        if self.client:
            self.sync_with_broker()

    def sync_with_broker(self):
        """Synchroniser positions et cash avec le broker"""
        if not self.client:
            return

        try:
            account = self.client.get_account()
            if account:
                self.cash = float(account.get('cash', 0))

            positions = self.client.get_positions()
            # Normaliser positions {USDT: 120, BTC: 0.5, ...}
            self.positions = {pos['symbol']: float(pos['qty']) for pos in positions}

            logger.info(f"🔄 Sync broker: ${self.cash:,.2f} cash, {len(self.positions)} positions")

        except Exception as e:
            logger.error(f"❌ Erreur sync broker: {e}")

    def get_market_data(self, days=60):
        """
        Récupérer les données de marché (plus de jours nécessaires pour features V6)
        """
        logger.info(f"📡 Récupération données ({days} jours)...")
        data = {}
        tickers_to_fetch = []
        
        # 1. Vérifier Cache
        for ticker in self.tickers:
            cache_file = f'data_cache/{ticker}.csv'
            if os.path.exists(cache_file):
                try:
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    if not df.empty:
                        last_date = pd.to_datetime(df.index[-1])
                        if last_date.tzinfo:
                            last_date = last_date.tz_localize(None)
                        
                        # Vérifier fraîcheur (max 1 jour ouvrable)
                        if (datetime.now() - last_date).days <= 2 and len(df) >= days:
                            data[ticker] = df.tail(days)
                            # logger.info(f"  ✅ {ticker}: Cache OK")
                            continue
                except Exception:
                    pass
            tickers_to_fetch.append(ticker)
        
        # 2. Fetch Manquant
        if tickers_to_fetch:
            logger.info(f"📥 Téléchargement de {len(tickers_to_fetch)} tickers...")
            if self.data_fetcher:
                try:
                    fetched = self.data_fetcher.fetch_multiple(tickers_to_fetch, days=days, save_cache=True)
                    data.update(fetched)
                except Exception as e:
                    logger.error(f"❌ Erreur fetch data: {e}")
            
            # Fallback yfinance si nécessaire
            remaining = [t for t in tickers_to_fetch if t not in data]
            if remaining:
                import yfinance as yf
                for t in remaining:
                    try:
                        end = datetime.now()
                        start = end - timedelta(days=days)
                        df = yf.download(t, start=start, end=end, interval='1d', progress=False)
                        if not df.empty:
                            if isinstance(df.columns, pd.MultiIndex):
                                df.columns = df.columns.get_level_values(0)
                            data[t] = df
                            df.to_csv(f'data_cache/{t}.csv')
                    except Exception as e:
                        logger.warning(f"⚠️ Échec yfinance pour {t}: {e}")

        # Nettoyage final
        valid_data = {k: v for k, v in data.items() if not v.empty and len(v) > 30}
        logger.info(f"✅ {len(valid_data)}/{len(self.tickers)} tickers prêts")
        return valid_data

    def get_predictions(self, data):
        """Générer prédictions avec le modèle V6"""
        logger.info("🔮 Analyse et Prédictions...")
        
        if not data:
            return {}
            
        try:
            # V6 a besoin d'historique pour calculer les features (RSI, Bollinger, etc.)
            # On passe tout le dataframe à l'environnement, qui gérera feature engineering
            
            # Créer env temporaire pour inférence
            # Note: Max steps petit car on fait juste un reset pour avoir l'état actuel
            env = UniversalTradingEnvV6BetterTiming(
                data=data,
                initial_balance=self.cash,
                max_steps=100, # Juste pour init
                buy_pct=0.2 # Config par défaut
            )
            
            # Reset pour s'aligner sur la fin des données (ou proche)
            # L'environnement V6 calcule tout à l'init
            # On doit forcer 'current_step' à la fin pour avoir la prédiction sur la dernière bougie
            
            # Hack: Accéder directement aux features pré-calculées
            # L'observation attendue par le modèle est celle du 'step' courant.
            # Pour le trading live, on veut l'obs correspondant au dernier point de données disponible.
            
            predictions = {}
            obs_dict = {} # Pour debugging si besoin
            
            # Pour chaque ticker, on veut savoir quelle action prendre "maintenant"
            # UniversalTradingEnv est prévu pour training (épisodique).
            # Pour l'inférence live, on simule un step à la fin du DF.
            
            # Force step to last available index
            max_idx = min(len(df) for df in env.processed_data.values()) - 1
            env.current_step = max_idx
            
            # Get observation for current state
            obs = env._get_observation()
            
            # Prédiction
            actions, _ = self.model.predict(obs, deterministic=True)
            
            # Convertir actions (Array MultiDiscrete)
            # [action_ticker1, action_ticker2, ...]
            
            action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            
            counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            
            for i, ticker in enumerate(env.tickers):
                if i < len(actions):
                    act_idx = actions[i]
                    decision = action_map.get(act_idx, 'HOLD')
                    
                    # Logique de filtrage basique post-prediction
                    current_pos = self.positions.get(ticker, 0)
                    
                    if decision == 'SELL' and current_pos <= 0:
                        decision = 'HOLD' # Rien à vendre
                    elif decision == 'BUY' and current_pos > 0:
                        decision = 'HOLD' # Déjà en position (éviter sur-exposition simple)
                    
                    predictions[ticker] = decision
                    counts[decision] += 1
            
            logger.info(f"📊 Résultat: {counts['BUY']} BUY | {counts['SELL']} SELL | {counts['HOLD']} HOLD")
            return predictions

        except Exception as e:
            logger.error(f"❌ Erreur prédictions: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def execute_trades(self, predictions, current_prices):
        """Exécuter les ordres"""
        logger.info("💼 Exécution des ordres...")
        
        for ticker, action in predictions.items():
            if action == 'HOLD':
                continue
                
            price = current_prices.get(ticker)
            if not price:
                continue
                
            logger.info(f"👉 Signal {action} sur {ticker} (Prix: ${price:.2f})")
            
            if self.client:
                # Mode Broker Réel / Paper
                try:
                    if action == 'BUY':
                        # Montant fixe ou % du cash
                        amount = 5000 # Exemple $5k par trade
                        qty = amount / price
                        if qty < 0.01: continue
                        
                        self.client.place_market_order(ticker, qty, 'buy')
                        logger.info(f"  ✅ Ordre BUY envoyé: {qty:.4f} {ticker}")
                        
                    elif action == 'SELL':
                        qty = self.positions.get(ticker, 0)
                        if qty > 0:
                            self.client.close_position(ticker)
                            logger.info(f"  ✅ Ordre SELL envoyé: {qty:.4f} {ticker}")
                except Exception as e:
                    logger.error(f"  ❌ Erreur ordre {ticker}: {e}")
            else:
                # Simulation locale simple dans les logs
                if action == 'BUY':
                    logger.info(f"  [SIM] BUY {ticker}")
                elif action == 'SELL':
                    logger.info(f"  [SIM] SELL {ticker}")

    def run_cycle(self):
        """Un cycle complet"""
        logger.info("\n" + "="*60)
        logger.info(f"⏰ Cycle {datetime.now().strftime('%H:%M:%S')}")
        
        # 1. Update Data
        data = self.get_market_data(days=100) # Besoin d'historique pour V6
        if not data:
            logger.warning("Pas de données, attente...")
            return

        # 2. Sync Broker
        self.sync_with_broker()
        
        # 3. Predict
        predictions = self.get_predictions(data)
        
        # 4. Execute
        current_prices = {t: df['Close'].iloc[-1] for t, df in data.items()}
        self.execute_trades(predictions, current_prices)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Ploutos Trader V6')
    parser.add_argument('--model', default='data/models/brain_tech.zip', help='Chemin modèle')
    parser.add_argument('--paper', action='store_true', help='Mode Paper Trading')
    parser.add_argument('--interval', type=int, default=15, help='Intervalle minutes')
    parser.add_argument('--broker', default='etoro', help='Broker (etoro/alpaca)')
    
    args = parser.parse_args()
    
    # Setup Env pour Broker (compatibilité)
    os.environ['BROKER'] = args.broker
    
    logger.info(f"🚀 Démarrage Trader V6")
    logger.info(f"Model: {args.model}")
    logger.info(f"Broker: {args.broker} (Paper: {args.paper})")
    
    bot = TraderV6(model_path=args.model, paper_trading=args.paper)
    
    try:
        while True:
            bot.run_cycle()
            logger.info(f"💤 Pause {args.interval} min...")
            time.sleep(args.interval * 60)
    except KeyboardInterrupt:
        logger.info("🛑 Arrêt demandé")

if __name__ == '__main__':
    main()
