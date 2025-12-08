#!/usr/bin/env python3
"""
🧪 BACKTEST DE FIABILITÉ

Teste la fiabilité du bot sur données historiques réelles

Ce script:
1. Charge le modèle entraîné
2. Simule 90 jours de trading
3. Calcule toutes les métriques
4. Compare vs Buy & Hold
5. Génère rapport détaillé

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

# Import modules Ploutos
from core.data_fetcher import UniversalDataFetcher
from core.universal_environment_v2 import UniversalTradingEnvV2
from stable_baselines3 import PPO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration - MEMES 10 TICKERS QUE L'ENTRAINEMENT
TICKERS = [
    # Tech Growth
    'NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN',
    # Indices
    'SPY', 'QQQ', 'VOO',
    # Sectoriels
    'XLE',  # Energy
    'XLF'   # Finance
]

INITIAL_BALANCE = 100000
BACKTEST_DAYS = 90
MODEL_PATH = "models/ppo_trading_v2_latest.zip"


class BacktestReliability:
    """Teste la fiabilité du bot"""
    
    def __init__(self, model_path, tickers, initial_balance=100000):
        self.model_path = Path(model_path)
        self.tickers = tickers
        self.initial_balance = initial_balance
        
        self.model = None
        self.env = None
        self.results = {}
        
    def load_model(self):
        """Charger le modèle entraîné"""
        if not self.model_path.exists():
            logger.error(f"❌ Modèle introuvable: {self.model_path}")
            return False
        
        try:
            self.model = PPO.load(self.model_path)
            logger.info(f"✅ Modèle chargé: {self.model_path.name}")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}")
            return False
    
    def fetch_data(self, days=90):
        """Charger données historiques"""
        logger.info(f"📊 Chargement données {days} jours...")
        
        fetcher = UniversalDataFetcher()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 100)  # +100 pour warmup
        
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
    
    def run_backtest(self, data, episodes=1):
        """Exécuter le backtest"""
        logger.info(f"\n🎮 BACKTEST EN COURS...")
        logger.info(f"  • Capital initial: ${self.initial_balance:,.2f}")
        logger.info(f"  • Tickers: {len(data)}")
        logger.info(f"  • Épisodes: {episodes}\n")
        
        # Créer environnement
        self.env = UniversalTradingEnvV2(
            data=data,
            initial_balance=self.initial_balance,
            commission=0.0001,
            max_steps=5000,
            buy_pct=0.2
        )
        
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
                
                portfolio_history.append(info['portfolio_value'])
                
                # Enregistrer trades
                if len(self.env.trades_history) > len(episode_trades):
                    new_trades = self.env.trades_history[len(episode_trades):]
                    episode_trades.extend(new_trades)
                
                step += 1
                
                if step % 100 == 0:
                    logger.info(f"  Step {step}: Portfolio ${info['portfolio_value']:,.2f}")
            
            # Résultats de l'épisode
            final_value = info['portfolio_value']
            total_return = (final_value - self.initial_balance) / self.initial_balance
            
            result = {
                'episode': ep + 1,
                'final_value': final_value,
                'total_return': total_return,
                'total_trades': len(episode_trades),
                'portfolio_history': portfolio_history,
                'trades': episode_trades
            }
            
            all_results.append(result)
            
            logger.info(f"\n✅ Épisode {ep + 1} terminé:")
            logger.info(f"  • Portfolio final: ${final_value:,.2f}")
            logger.info(f"  • Return: {total_return:.2%}")
            logger.info(f"  • Trades: {len(episode_trades)}")
        
        return all_results
    
    def calculate_metrics(self, results):
        """Calculer métriques de performance"""
        logger.info(f"\n📈 CALCUL MÉTRIQUES...")
        
        # Moyenne sur tous les épisodes
        avg_final_value = np.mean([r['final_value'] for r in results])
        avg_return = np.mean([r['total_return'] for r in results])
        
        # Trades
        all_trades = []
        for r in results:
            all_trades.extend(r['trades'])
        
        # Win rate
        wins = [t for t in all_trades if 'pnl' in t and t['pnl'] > 0]
        losses = [t for t in all_trades if 'pnl' in t and t['pnl'] < 0]
        win_rate = len(wins) / (len(wins) + len(losses)) if (wins or losses) else 0
        
        # PnL moyen
        if wins:
            avg_win = np.mean([t['pnl'] for t in wins])
        else:
            avg_win = 0
        
        if losses:
            avg_loss = np.mean([abs(t['pnl']) for t in losses])
        else:
            avg_loss = 0
        
        # Profit factor
        total_wins = sum([t['pnl'] for t in wins])
        total_losses = abs(sum([t['pnl'] for t in losses]))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Sharpe ratio (simplifié)
        returns = [r['total_return'] for r in results]
        sharpe = (np.mean(returns) / np.std(returns)) if len(returns) > 1 else 0
        
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
            'total_trades': len(all_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd
        }
        
        return metrics
    
    def compare_buy_hold(self, data):
        """Comparer vs stratégie Buy & Hold"""
        logger.info(f"\n📊 COMPARAISON BUY & HOLD...")
        
        # Investir équitablement dans tous les tickers
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
    
    def generate_report(self, results, metrics, buy_hold):
        """Générer rapport complet"""
        logger.info(f"\n📄 GÉNÉRATION RAPPORT...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'initial_balance': self.initial_balance,
                'tickers': self.tickers,
                'model': str(self.model_path),
                'episodes': len(results)
            },
            'performance': {
                'bot': {
                    'final_value': metrics['avg_final_value'],
                    'return': metrics['avg_return'],
                    'win_rate': metrics['win_rate'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown': metrics['max_drawdown'],
                    'total_trades': metrics['total_trades'],
                    'profit_factor': metrics['profit_factor']
                },
                'buy_hold': buy_hold,
                'outperformance': metrics['avg_return'] - buy_hold['return']
            },
            'reliability_score': self._calculate_reliability_score(metrics)
        }
        
        # Sauvegarder
        output_file = Path('logs/backtest_reliability.json')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Rapport sauvegardé: {output_file}")
        
        return report
    
    def _calculate_reliability_score(self, metrics):
        """Score de fiabilité (0-100)"""
        score = 0
        
        # Win rate (40 points)
        score += min(metrics['win_rate'] * 80, 40)
        
        # Sharpe (30 points)
        score += min(max(metrics['sharpe_ratio'], 0) * 20, 30)
        
        # Drawdown (20 points)
        score += max(0, 20 - metrics['max_drawdown'] * 100)
        
        # Profit factor (10 points)
        score += min(metrics['profit_factor'] * 5, 10)
        
        return min(score, 100)
    
    def print_report(self, report):
        """Afficher rapport"""
        print("\n" + "="*70)
        print("📄 RAPPORT DE FIABILITÉ")
        print("="*70)
        
        bot = report['performance']['bot']
        bh = report['performance']['buy_hold']
        
        print(f"\n🤖 PERFORMANCE DU BOT:")
        print(f"  • Portfolio final: ${bot['final_value']:,.2f}")
        print(f"  • Return: {bot['return']:.2%}")
        print(f"  • Win Rate: {bot['win_rate']:.1%}")
        print(f"  • Sharpe Ratio: {bot['sharpe_ratio']:.2f}")
        print(f"  • Max Drawdown: {bot['max_drawdown']:.1%}")
        print(f"  • Profit Factor: {bot['profit_factor']:.2f}")
        print(f"  • Total Trades: {bot['total_trades']}")
        
        print(f"\n📈 BUY & HOLD:")
        print(f"  • Portfolio final: ${bh['final_value']:,.2f}")
        print(f"  • Return: {bh['return']:.2%}")
        
        print(f"\n🏆 COMPARAISON:")
        outperf = report['performance']['outperformance']
        if outperf > 0:
            print(f"  ✅ Bot surperforme de {outperf:.2%}")
        else:
            print(f"  ❌ Bot sous-performe de {abs(outperf):.2%}")
        
        print(f"\n🎯 SCORE DE FIABILITÉ: {report['reliability_score']:.1f}/100")
        
        if report['reliability_score'] >= 70:
            print("  ✅ EXCELLENT - Bot fiable")
        elif report['reliability_score'] >= 50:
            print("  🟡 MOYEN - Améliorations possibles")
        else:
            print("  ❌ FAIBLE - Re-entraînement recommandé")
        
        print("\n" + "="*70 + "\n")


def main():
    print("\n" + "="*70)
    print("🧪 BACKTEST DE FIABILITÉ - PLOUTOS")
    print("="*70 + "\n")
    
    # Initialiser
    backtest = BacktestReliability(
        model_path=MODEL_PATH,
        tickers=TICKERS,
        initial_balance=INITIAL_BALANCE
    )
    
    # Charger modèle
    if not backtest.load_model():
        logger.error("❌ Impossible de charger le modèle")
        return
    
    # Charger données
    data = backtest.fetch_data(days=BACKTEST_DAYS)
    if data is None:
        return
    
    # Exécuter backtest
    results = backtest.run_backtest(data, episodes=1)
    
    # Calculer métriques
    metrics = backtest.calculate_metrics(results)
    
    # Buy & Hold
    buy_hold = backtest.compare_buy_hold(data)
    
    # Générer rapport
    report = backtest.generate_report(results, metrics, buy_hold)
    
    # Afficher
    backtest.print_report(report)
    
    print("✅ Backtest terminé !\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption utilisateur\n")
    except Exception as e:
        logger.error(f"❌ Erreur: {e}", exc_info=True)
