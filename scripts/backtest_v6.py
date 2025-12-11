#!/usr/bin/env python3
"""🧪 BACKTEST V6 - Pour modèles avec Features V2

Utilise UniversalTradingEnvV6BetterTiming (85 features/ticker)
Compatible avec modèles 1293 dimensions
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

from core.data_fetcher import UniversalDataFetcher
from stable_baselines3 import PPO
from core.universal_environment_v6_better_timing import UniversalTradingEnvV6BetterTiming

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ✅ Configuration - MEMES 15 TICKERS
TICKERS_V6 = [
    'NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'TSLA',
    'SPY', 'QQQ', 'VOO', 'VTI',
    'XLE', 'XLF', 'XLK', 'XLV'
]

INITIAL_BALANCE = 100000
BACKTEST_DAYS = 90


class BacktestV6:
    """
    Backtest pour modèles V6 (Features V2 - 85 features/ticker)
    """
    
    def __init__(self, model_path, tickers, initial_balance=100000):
        self.model_path = Path(model_path)
        self.tickers = tickers
        self.initial_balance = initial_balance
        
        self.model = None
        self.env = None
        self.results = {}
        
    def load_model(self):
        """✅ Charger le modèle"""
        if not self.model_path.exists():
            logger.error(f"❌ Modèle introuvable: {self.model_path}")
            return False
        
        try:
            self.model = PPO.load(self.model_path)
            logger.info(f"✅ Modèle chargé: {self.model_path.name}")
            logger.info(f"  • Observation space: {self.model.observation_space.shape}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def fetch_data(self, days=90):
        """Charger données historiques"""
        logger.info(f"📊 Chargement données {days} jours...")
        logger.info(f"  • Tickers: {len(self.tickers)}")
        
        fetcher = UniversalDataFetcher()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 100)
        
        data = {}
        failed = []
        
        for ticker in self.tickers:
            try:
                df = fetcher.fetch(
                    ticker,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    interval='1h'
                )
                
                if df is not None and len(df) > 100:
                    data[ticker] = df
                    logger.info(f"  ✅ {ticker}: {len(df)} barres")
                else:
                    failed.append(ticker)
                    logger.warning(f"  ⚠️  {ticker}: données insuffisantes")
                    
            except Exception as e:
                failed.append(ticker)
                logger.error(f"  ❌ {ticker}: {e}")
        
        if failed:
            logger.warning(f"⚠️  {len(failed)} ticker(s) échoué(s): {failed}")
        
        if len(data) < 5:
            logger.error("❌ Pas assez de données pour backtest")
            return None
        
        logger.info(f"✅ {len(data)} tickers chargés")
        return data
    
    def create_environment(self, data):
        """✅ Créer environnement V6 BetterTiming"""
        logger.info(f"\n🏭 Création environnement V6 BetterTiming...")
        
        self.env = UniversalTradingEnvV6BetterTiming(
            data=data,
            initial_balance=self.initial_balance,
            commission=0.0,
            sec_fee=0.0000221,
            finra_taf=0.000145,
            max_steps=5000,
            buy_pct=0.2,
            slippage_model='realistic',
            spread_bps=2.0,
            max_position_pct=0.25,
            max_trades_per_day=10,
            min_holding_period=2,
            reward_scaling=1.5,
            use_sharpe_penalty=True,
            use_drawdown_penalty=True,
            reward_trade_success=0.5,
            penalty_overtrading=0.005,
            drawdown_penalty_factor=3.0
        )
        
        logger.info(f"✅ Environnement créé")
        
        # Vérifier taille observation
        obs, _ = self.env.reset()
        logger.info(f"  • Observation shape: {obs.shape}")
        logger.info(f"  • Model expects: {self.model.observation_space.shape}")
        
        if obs.shape[0] != self.model.observation_space.shape[0]:
            logger.error(f"❌ MISMATCH! Env génère {obs.shape[0]} dims, model attend {self.model.observation_space.shape[0]}")
            raise ValueError("Incompatibilité taille observation")
        
        logger.info(f"  ✅ MATCH PARFAIT!")
        
        return self.env
    
    def run_backtest(self, data, episodes=1):
        """Exécuter le backtest"""
        logger.info(f"\n🎮 BACKTEST EN COURS...")
        logger.info(f"  • Capital initial: ${self.initial_balance:,.2f}")
        logger.info(f"  • Tickers: {len(data)}")
        logger.info(f"  • Épisodes: {episodes}")
        logger.info(f"  • Version: V6_BETTER_TIMING\n")
        
        # Créer environnement
        self.env = self.create_environment(data)
        
        all_results = []
        
        for ep in range(episodes):
            logger.info(f"\n🔄 Épisode {ep + 1}/{episodes}")
            
            obs, info = self.env.reset()
            done = False
            truncated = False
            step = 0
            
            episode_trades = []
            portfolio_history = [self.initial_balance]
            
            while not (done or truncated):
                # Prédiction du modèle
                action, _states = self.model.predict(obs, deterministic=True)
                
                # Step
                obs, reward, done, truncated, info = self.env.step(action)
                
                pf_value = info.get('equity', self.initial_balance)
                portfolio_history.append(pf_value)
                
                step += 1
                
                if step % 100 == 0:
                    logger.info(f"  Step {step}: Portfolio ${pf_value:,.2f}")
            
            # Résultats de l'épisode
            final_value = portfolio_history[-1]
            total_return = (final_value - self.initial_balance) / self.initial_balance
            
            result = {
                'episode': ep + 1,
                'final_value': final_value,
                'total_return': total_return,
                'total_trades': info.get('total_trades', 0),
                'winning_trades': info.get('winning_trades', 0),
                'losing_trades': info.get('losing_trades', 0),
                'portfolio_history': portfolio_history
            }
            
            all_results.append(result)
            
            logger.info(f"\n✅ Épisode {ep + 1} terminé:")
            logger.info(f"  • Portfolio final: ${final_value:,.2f}")
            logger.info(f"  • Return: {total_return:.2%}")
            logger.info(f"  • Trades: {result['total_trades']}")
            logger.info(f"  • Win Rate: {result['winning_trades']/(result['winning_trades']+result['losing_trades'])*100:.1f}%" if (result['winning_trades']+result['losing_trades']) > 0 else "  • Win Rate: N/A")
        
        return all_results
    
    def calculate_metrics(self, results):
        """Calculer métriques de performance"""
        logger.info(f"\n📈 CALCUL MÉTRIQUES...")
        
        avg_final_value = np.mean([r['final_value'] for r in results])
        avg_return = np.mean([r['total_return'] for r in results])
        
        total_trades = sum([r['total_trades'] for r in results])
        total_wins = sum([r['winning_trades'] for r in results])
        total_losses = sum([r['losing_trades'] for r in results])
        
        win_rate = total_wins / (total_wins + total_losses) if (total_wins + total_losses) > 0 else 0
        
        # Sharpe ratio simplifié
        returns = [r['total_return'] for r in results]
        sharpe = (np.mean(returns) / np.std(returns)) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        # Max drawdown
        max_dd = 0
        for r in results:
            history = r['portfolio_history']
            cumulative = np.array(history)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (running_max - cumulative) / running_max
            max_dd = max(max_dd, np.max(drawdown))
        
        metrics = {
            'avg_final_value': avg_final_value,
            'avg_return': avg_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd
        }
        
        return metrics
    
    def compare_buy_hold(self, data):
        """Comparer vs stratégie Buy & Hold"""
        logger.info(f"\n📊 COMPARAISON BUY & HOLD...")
        
        amount_per_ticker = self.initial_balance / len(data)
        
        buy_hold_values = []
        
        for ticker, df in data.items():
            if len(df) < 2:
                continue
            
            initial_price = df['Close'].iloc[0]
            final_price = df['Close'].iloc[-1]
            
            shares = amount_per_ticker / initial_price
            final_value = shares * final_price
            
            buy_hold_values.append(final_value)
        
        total_buy_hold = sum(buy_hold_values)
        buy_hold_return = (total_buy_hold - self.initial_balance) / self.initial_balance
        
        return {
            'final_value': total_buy_hold,
            'return': buy_hold_return
        }
    
    def print_report(self, results, metrics, buy_hold):
        """Afficher rapport"""
        print("\n" + "="*70)
        print(f"📄 RAPPORT BACKTEST V6 - BETTER TIMING")
        print("="*70)
        
        print(f"\n🤖 PERFORMANCE DU BOT:")
        print(f"  • Portfolio final: ${metrics['avg_final_value']:,.2f}")
        print(f"  • Return: {metrics['avg_return']:.2%}")
        print(f"  • Win Rate: {metrics['win_rate']:.1%}")
        print(f"  • Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  • Max Drawdown: {metrics['max_drawdown']:.1%}")
        print(f"  • Total Trades: {metrics['total_trades']}")
        
        print(f"\n📈 BUY & HOLD:")
        print(f"  • Portfolio final: ${buy_hold['final_value']:,.2f}")
        print(f"  • Return: {buy_hold['return']:.2%}")
        
        print(f"\n🏆 COMPARAISON:")
        outperf = metrics['avg_return'] - buy_hold['return']
        if outperf > 0:
            print(f"  ✅ Bot surperforme de {outperf:.2%}")
        else:
            print(f"  ❌ Bot sous-performe de {abs(outperf):.2%}")
        
        print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtest V6 Better Timing')
    parser.add_argument('--model', type=str, required=True, help='Chemin du modèle')
    parser.add_argument('--days', type=int, default=90, help='Nombre de jours')
    parser.add_argument('--episodes', type=int, default=5, help="Nombre d'épisodes")
    args = parser.parse_args()
    
    backtest = BacktestV6(
        model_path=args.model,
        tickers=TICKERS_V6,
        initial_balance=INITIAL_BALANCE
    )
    
    if not backtest.load_model():
        sys.exit(1)
    
    data = backtest.fetch_data(days=args.days)
    if data is None:
        sys.exit(1)
    
    results = backtest.run_backtest(data, episodes=args.episodes)
    metrics = backtest.calculate_metrics(results)
    buy_hold = backtest.compare_buy_hold(data)
    
    backtest.print_report(results, metrics, buy_hold)
    
    print("✅ Backtest terminé !\n")
