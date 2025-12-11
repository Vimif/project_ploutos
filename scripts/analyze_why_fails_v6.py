#!/usr/bin/env python3
"""🔍 ANALYSE APPROFONDIE V6 : Pourquoi l'IA échoue vs Buy & Hold

Version V6 pour modèles avec Features V2 (85 features/ticker)

Analyses:
1. Distribution temporelle des actions
2. Qualité du timing (buy high? sell low?)
3. Opportunités manquées
4. Patterns d'échec
5. Comparison action IA vs signaux marché

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json

from stable_baselines3 import PPO
from core.data_fetcher import UniversalDataFetcher
from core.universal_environment_v6_better_timing import UniversalTradingEnvV6BetterTiming

# Config
TICKERS = [
    'NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'TSLA',
    'SPY', 'QQQ', 'VOO', 'VTI',
    'XLE', 'XLF', 'XLK', 'XLV'
]

INITIAL_BALANCE = 100000


class WhyFailsAnalyzerV6:
    """🔍 Analyse approfondie des échecs - Version V6"""
    
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = None
        self.env = None
        self.data = None
        
        self.analysis_results = {
            'actions_log': [],
            'market_moves': {},
            'timing_quality': {},
            'missed_opportunities': [],
            'failure_patterns': []
        }
    
    def load_model(self):
        print(f"\n📦 Chargement modèle: {self.model_path.name}")
        self.model = PPO.load(self.model_path)
        print(f"✅ Modèle chargé")
        return True
    
    def load_data(self, days=90):
        print(f"\n📊 Chargement données ({days} jours)...")
        
        fetcher = UniversalDataFetcher()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 50)
        
        data = {}
        for ticker in TICKERS:
            try:
                df = fetcher.fetch(
                    ticker,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    interval='1h'
                )
                if df is not None and len(df) > 50:
                    data[ticker] = df
                    print(f"  ✅ {ticker}: {len(df)} barres")
            except Exception as e:
                print(f"  ❌ {ticker}: {e}")
        
        self.data = data
        print(f"\n✅ {len(data)} tickers chargés")
        return data
    
    def create_environment(self):
        print(f"\n🏭 Création environnement V6 BetterTiming...")
        
        self.env = UniversalTradingEnvV6BetterTiming(
            data=self.data,
            initial_balance=INITIAL_BALANCE,
            commission=0.0,
            sec_fee=0.0000221,
            finra_taf=0.000145,
            max_steps=1000,
            buy_pct=0.2,
            slippage_model='realistic',
            spread_bps=2.0,
            max_position_pct=0.25,
            max_trades_per_day=10,
            min_holding_period=2
        )
        
        print(f"✅ Environnement créé")
        return self.env
    
    def analyze_market_conditions(self):
        """✅ Analyser conditions de marché"""
        print(f"\n📈 Analyse conditions marché...")
        
        market_analysis = {}
        
        for ticker, df in self.data.items():
            if len(df) < 2:
                continue
            
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252 * 6.5)
            
            first_price = df['Close'].iloc[0]
            last_price = df['Close'].iloc[-1]
            total_return = (last_price - first_price) / first_price
            
            big_ups = (returns > 0.03).sum()
            big_downs = (returns < -0.03).sum()
            
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()
            
            market_analysis[ticker] = {
                'total_return': total_return,
                'volatility': volatility,
                'big_ups': int(big_ups),
                'big_downs': int(big_downs),
                'max_drawdown': max_dd,
                'trend': 'UP' if total_return > 0.05 else 'DOWN' if total_return < -0.05 else 'FLAT'
            }
            
            print(f"  {ticker:5s}: {total_return:+6.1%} | Vol: {volatility:4.1%} | "
                  f"Trend: {market_analysis[ticker]['trend']:5s} | "
                  f"Big moves: {big_ups}↑ {big_downs}↓")
        
        self.analysis_results['market_conditions'] = market_analysis
        return market_analysis
    
    def run_episode_with_logging(self, deterministic=True):
        """✅ Exécuter episode en loggant TOUT"""
        mode = "DETERMINISTIC" if deterministic else "STOCHASTIC"
        print(f"\n🎮 Exécution episode ({mode})...")
        
        obs, _ = self.env.reset()
        done = False
        truncated = False
        step = 0
        
        actions_log = []
        portfolio_history = [INITIAL_BALANCE]
        
        while not (done or truncated) and step < 500:
            action, _states = self.model.predict(obs, deterministic=deterministic)
            
            current_prices = {}
            for i, ticker in enumerate(TICKERS[:len(action)]):
                if ticker in self.data:
                    df = self.data[ticker]
                    current_step = self.env.current_step
                    if current_step < len(df):
                        current_prices[ticker] = df.iloc[current_step]['Close']
            
            obs, reward, done, truncated, info = self.env.step(action)
            
            pf_value = info.get('equity', INITIAL_BALANCE)
            portfolio_history.append(pf_value)
            
            for i, (ticker, act) in enumerate(zip(TICKERS[:len(action)], action)):
                if act != 0 and ticker in current_prices:
                    actions_log.append({
                        'step': step,
                        'ticker': ticker,
                        'action': 'BUY' if act == 1 else 'SELL',
                        'price': current_prices[ticker],
                        'portfolio_value': pf_value
                    })
            
            step += 1
            
            if step % 100 == 0:
                print(f"  Step {step}: ${pf_value:,.2f} | Actions: {len([a for a in action if a != 0])}")
        
        final_value = portfolio_history[-1]
        total_return = (final_value - INITIAL_BALANCE) / INITIAL_BALANCE
        
        print(f"\n✅ Episode terminé:")
        print(f"  • Final value: ${final_value:,.2f}")
        print(f"  • Return: {total_return:+.2%}")
        print(f"  • Actions exécutées: {len(actions_log)}")
        
        self.analysis_results['actions_log'] = actions_log
        self.analysis_results['portfolio_history'] = portfolio_history
        self.analysis_results['final_return'] = total_return
        
        return actions_log, portfolio_history
    
    def analyze_timing_quality(self, actions_log):
        """✅ Analyser qualité du timing des trades"""
        print(f"\n🎯 Analyse qualité du timing...")
        
        if len(actions_log) == 0:
            print("❌ Aucune action à analyser (0 trades)")
            print("\n💡 CAUSE #1: MODÈLE NE TRADE PAS")
            print("   → Le modèle a appris que HOLD = sécurité")
            print("   → Rewards/pénalités punissent trop l'action")
            print("   → Entropy trop bas pour exploration")
            return {'no_trades': True}
        
        timing_analysis = {
            'buy_high': 0,
            'buy_low': 0,
            'sell_high': 0,
            'sell_low': 0,
            'good_buys': [],
            'bad_buys': [],
            'good_sells': [],
            'bad_sells': []
        }
        
        for action in actions_log:
            ticker = action['ticker']
            if ticker not in self.data:
                continue
            
            df = self.data[ticker]
            action_price = action['price']
            action_step = action['step']
            future_start = action_step + 1
            future_end = min(action_step + 11, len(df))
            
            if future_end <= future_start:
                continue
            
            future_prices = df.iloc[future_start:future_end]['Close'].values
            avg_future = np.mean(future_prices)
            
            if action['action'] == 'BUY':
                if avg_future > action_price * 1.01:
                    timing_analysis['buy_low'] += 1
                    timing_analysis['good_buys'].append(action)
                else:
                    timing_analysis['buy_high'] += 1
                    timing_analysis['bad_buys'].append(action)
            
            elif action['action'] == 'SELL':
                if avg_future < action_price * 0.99:
                    timing_analysis['sell_high'] += 1
                    timing_analysis['good_sells'].append(action)
                else:
                    timing_analysis['sell_low'] += 1
                    timing_analysis['bad_sells'].append(action)
        
        total_buys = timing_analysis['buy_high'] + timing_analysis['buy_low']
        total_sells = timing_analysis['sell_high'] + timing_analysis['sell_low']
        
        print(f"\n  📈 BUYs:")
        if total_buys > 0:
            buy_quality = timing_analysis['buy_low'] / total_buys
            print(f"    ✅ Good (buy low):  {timing_analysis['buy_low']} ({buy_quality*100:.1f}%)")
            print(f"    ❌ Bad (buy high):  {timing_analysis['buy_high']} ({timing_analysis['buy_high']/total_buys*100:.1f}%)")
            
            if buy_quality < 0.4:
                print(f"\n❌ CAUSE #2: MAUVAIS TIMING D'ACHAT ({buy_quality:.0%} good)")
                print("   → Modèle achète quand prix déjà haut")
                print("   → Features momentum/trend inefficaces")
            elif buy_quality > 0.5:
                print(f"\n✅ EXCELLENT TIMING D'ACHAT! ({buy_quality:.0%} good)")
                print("   → Features V2 fonctionnent bien")
                print("   → Support/resistance efficaces")
        else:
            print("    Aucun BUY")
        
        print(f"\n  📉 SELLs:")
        if total_sells > 0:
            sell_quality = timing_analysis['sell_high'] / total_sells
            print(f"    ✅ Good (sell high): {timing_analysis['sell_high']} ({sell_quality*100:.1f}%)")
            print(f"    ❌ Bad (sell low):  {timing_analysis['sell_low']} ({timing_analysis['sell_low']/total_sells*100:.1f}%)")
            
            if sell_quality < 0.4:
                print(f"\n❌ CAUSE #3: MAUVAIS TIMING DE VENTE ({sell_quality:.0%} good)")
                print("   → Modèle vend trop tôt ou trop tard")
            elif sell_quality > 0.5:
                print(f"\n✅ BON TIMING DE VENTE ({sell_quality:.0%} good)")
        else:
            print("    Aucun SELL")
        
        self.analysis_results['timing_quality'] = timing_analysis
        return timing_analysis
    
    def identify_missed_opportunities(self):
        """✅ Identifier opportunités manquées"""
        print(f"\n🔍 Recherche opportunités manquées...")
        
        missed = []
        
        for ticker, df in self.data.items():
            if len(df) < 20:
                continue
            
            returns = df['Close'].pct_change()
            big_moves = returns[abs(returns) > 0.05]
            
            if len(big_moves) > 0:
                missed.append({
                    'ticker': ticker,
                    'big_moves_count': len(big_moves),
                    'avg_move': big_moves.mean(),
                    'max_move': big_moves.max(),
                    'min_move': big_moves.min()
                })
        
        missed.sort(key=lambda x: x['big_moves_count'], reverse=True)
        
        print(f"\n  Top 5 tickers avec grandes moves:")
        for i, opp in enumerate(missed[:5]):
            print(f"    {i+1}. {opp['ticker']:5s}: {opp['big_moves_count']:2d} moves (avg: {opp['avg_move']:+.1%}, max: {opp['max_move']:+.1%})")
        
        total_big_moves = sum(o['big_moves_count'] for o in missed)
        actions_count = len(self.analysis_results.get('actions_log', []))
        
        if total_big_moves > 10:
            capture_rate = actions_count / total_big_moves if total_big_moves > 0 else 0
            print(f"\n📊 Statistiques opportunités:")
            print(f"   • Marché avait {total_big_moves} grandes moves")
            print(f"   • IA a fait {actions_count} actions")
            print(f"   • Ratio capture: {capture_rate*100:.1f}%")
            
            if capture_rate < 0.2:
                print(f"\n❌ CAUSE #4: OPPORTUNITÉS MANQUÉES")
                print(f"   → IA trop conservatrice")
            elif capture_rate > 0.5:
                print(f"\n✅ BONNE CAPTURE D'OPPORTUNITÉS")
        
        self.analysis_results['missed_opportunities'] = missed
        return missed
    
    def generate_final_report(self):
        """✅ Générer rapport final"""
        print("\n" + "="*70)
        print("📄 RAPPORT FINAL V6 : Pourquoi l'IA échoue vs Buy & Hold")
        print("="*70)
        
        print(f"\n1️⃣  PERFORMANCE:")
        final_return = self.analysis_results.get('final_return', 0)
        print(f"   IA Return: {final_return:+.2%}")
        
        market_cond = self.analysis_results.get('market_conditions', {})
        if market_cond:
            avg_market_return = np.mean([m['total_return'] for m in market_cond.values()])
            print(f"   Buy & Hold (avg): {avg_market_return:+.2%}")
            print(f"   Différence: {final_return - avg_market_return:+.2%}")
        
        print(f"\n2️⃣  ACTIVITÉ TRADING:")
        actions_count = len(self.analysis_results.get('actions_log', []))
        print(f"   Trades exécutés: {actions_count}")
        
        if actions_count == 0:
            print("   ⚠️  PROBLÈME MAJEUR: Aucun trade !")
        
        print(f"\n3️⃣  QUALITÉ TIMING:")
        timing = self.analysis_results.get('timing_quality', {})
        if timing.get('no_trades'):
            print("   N/A (pas de trades)")
        else:
            total_buys = timing.get('buy_high', 0) + timing.get('buy_low', 0)
            total_sells = timing.get('sell_high', 0) + timing.get('sell_low', 0)
            
            if total_buys > 0:
                buy_quality = timing.get('buy_low', 0) / total_buys
                print(f"   BUY quality: {buy_quality:.0%} ({timing.get('buy_low', 0)}/{total_buys} good)")
            
            if total_sells > 0:
                sell_quality = timing.get('sell_high', 0) / total_sells
                print(f"   SELL quality: {sell_quality:.0%} ({timing.get('sell_high', 0)}/{total_sells} good)")
        
        print("\n" + "="*70 + "\n")
        
        output_file = Path('logs/why_fails_analysis_v6.json')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json_results = {}
            for k, v in self.analysis_results.items():
                if isinstance(v, (list, dict, str, int, float, bool)):
                    json_results[k] = v
            json.dump(json_results, f, indent=2, default=str)
        
        print(f"✅ Rapport détaillé sauvegardé: {output_file}")
    
    def run_full_analysis(self):
        """Exécuter analyse complète"""
        print("\n" + "="*70)
        print("🔍 ANALYSE APPROFONDIE V6 : Pourquoi l'IA échoue")
        print("="*70)
        
        if not self.load_model():
            return
        
        data = self.load_data(days=90)
        if len(data) < 5:
            print("❌ Pas assez de données")
            return
        
        self.analyze_market_conditions()
        self.create_environment()
        actions_log, portfolio_history = self.run_episode_with_logging(deterministic=True)
        self.analyze_timing_quality(actions_log)
        self.identify_missed_opportunities()
        self.generate_final_report()
        
        print("✅ Analyse terminée !\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyse V6 pourquoi IA échoue')
    parser.add_argument('--model', type=str, required=True, help='Chemin du modèle V6')
    args = parser.parse_args()
    
    analyzer = WhyFailsAnalyzerV6(model_path=args.model)
    analyzer.run_full_analysis()
