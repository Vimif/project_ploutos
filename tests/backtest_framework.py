#!/usr/bin/env python3
"""
🧪 PLOUTOS BACKTESTING FRAMEWORK

Framework complet pour tester et comparer les modèles de trading

Features:
- Backtesting sur données historiques
- Comparaison PPO vs V7 vs PPO+V7
- Métriques de performance détaillées
- Rapports visuels et JSON
- Simulation réaliste des coûts

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO

try:
    from src.models.v7_predictor import V7Predictor
    V7_AVAILABLE = True
except:
    V7_AVAILABLE = False

try:
    from core.universal_environment_v2 import UniversalTradingEnvV2
except:
    from core.universal_environment import UniversalTradingEnv as UniversalTradingEnvV2


class BacktestFramework:
    """
    Framework de backtesting pour évaluer les modèles de trading
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.0005):
        """
        Args:
            initial_capital: Capital initial
            commission: Frais de transaction (0.1% par défaut)
            slippage: Slippage estimé (0.05% par défaut)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Résultats
        self.results = {}
        
    def load_historical_data(self, 
                            tickers: List[str], 
                            start_date: str,
                            end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Charge les données historiques pour le backtest
        
        Args:
            tickers: Liste des tickers
            start_date: Date de début (YYYY-MM-DD)
            end_date: Date de fin (YYYY-MM-DD)
        """
        print(f"\n📡 Chargement données historiques...")
        print(f"   Période: {start_date} à {end_date}")
        print(f"   Tickers: {', '.join(tickers)}")
        
        data = {}
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if not df.empty and len(df) > 100:
                    data[ticker] = df
                    print(f"   ✅ {ticker}: {len(df)} jours")
                else:
                    print(f"   ⚠️  {ticker}: Données insuffisantes")
            except Exception as e:
                print(f"   ❌ {ticker}: {e}")
        
        print(f"\n✅ {len(data)}/{len(tickers)} tickers chargés\n")
        return data
    
    def backtest_ppo_only(self, 
                          model_path: str,
                          data: Dict[str, pd.DataFrame],
                          test_split: float = 0.3) -> Dict:
        """
        Backtest avec PPO uniquement
        """
        print("\n" + "="*70)
        print("🤖 BACKTEST: PPO ONLY")
        print("="*70)
        
        # Charger modèle PPO
        model = PPO.load(model_path)
        print(f"✅ Modèle PPO chargé: {model_path}")
        
        # Split train/test
        test_data = self._split_data(data, test_split, phase='test')
        
        # Simuler trading
        trades, portfolio_values = self._simulate_trading(
            model=model,
            data=test_data,
            use_v7_filter=False
        )
        
        # Calculer métriques
        metrics = self._calculate_metrics(trades, portfolio_values)
        
        self.results['ppo_only'] = {
            'trades': trades,
            'portfolio_values': portfolio_values,
            'metrics': metrics
        }
        
        self._print_metrics("PPO Only", metrics)
        return metrics
    
    def backtest_v7_only(self,
                        data: Dict[str, pd.DataFrame],
                        test_split: float = 0.3) -> Dict:
        """
        Backtest avec V7 Enhanced uniquement
        """
        print("\n" + "="*70)
        print("⭐ BACKTEST: V7 ENHANCED ONLY")
        print("="*70)
        
        if not V7_AVAILABLE:
            print("❌ V7 non disponible")
            return {}
        
        # Charger V7
        v7 = V7Predictor()
        if not v7.load("momentum"):
            print("❌ Erreur chargement V7")
            return {}
        
        print("✅ V7 Enhanced chargé (68.35% accuracy)")
        
        # Split train/test
        test_data = self._split_data(data, test_split, phase='test')
        
        # Simuler trading avec V7
        trades, portfolio_values = self._simulate_v7_trading(v7, test_data)
        
        # Calculer métriques
        metrics = self._calculate_metrics(trades, portfolio_values)
        
        self.results['v7_only'] = {
            'trades': trades,
            'portfolio_values': portfolio_values,
            'metrics': metrics
        }
        
        self._print_metrics("V7 Enhanced Only", metrics)
        return metrics
    
    def backtest_ppo_plus_v7(self,
                             model_path: str,
                             data: Dict[str, pd.DataFrame],
                             test_split: float = 0.3) -> Dict:
        """
        Backtest avec PPO + V7 Enhanced (système hybrid)
        """
        print("\n" + "="*70)
        print("🚀 BACKTEST: PPO + V7 ENHANCED (HYBRID)")
        print("="*70)
        
        # Charger modèles
        model = PPO.load(model_path)
        print(f"✅ Modèle PPO chargé")
        
        v7 = None
        if V7_AVAILABLE:
            v7 = V7Predictor()
            if v7.load("momentum"):
                print("✅ V7 Enhanced chargé (filtre actif)")
            else:
                v7 = None
        
        if not v7:
            print("⚠️  V7 non disponible - utilise PPO only")
            return self.backtest_ppo_only(model_path, data, test_split)
        
        # Split train/test
        test_data = self._split_data(data, test_split, phase='test')
        
        # Simuler trading avec filtre V7
        trades, portfolio_values = self._simulate_trading(
            model=model,
            data=test_data,
            use_v7_filter=True,
            v7_predictor=v7
        )
        
        # Calculer métriques
        metrics = self._calculate_metrics(trades, portfolio_values)
        
        self.results['ppo_plus_v7'] = {
            'trades': trades,
            'portfolio_values': portfolio_values,
            'metrics': metrics
        }
        
        self._print_metrics("PPO + V7 Enhanced", metrics)
        return metrics
    
    def _split_data(self, data: Dict[str, pd.DataFrame], 
                   test_split: float, phase: str) -> Dict[str, pd.DataFrame]:
        """
        Split les données en train/test
        """
        split_data = {}
        
        for ticker, df in data.items():
            split_idx = int(len(df) * (1 - test_split))
            
            if phase == 'test':
                split_data[ticker] = df.iloc[split_idx:].copy()
            else:
                split_data[ticker] = df.iloc[:split_idx].copy()
        
        return split_data
    
    def _simulate_trading(self, 
                         model,
                         data: Dict[str, pd.DataFrame],
                         use_v7_filter: bool = False,
                         v7_predictor = None) -> Tuple[List[Dict], List[float]]:
        """
        Simule le trading avec le modèle
        """
        print(f"\n🔄 Simulation trading...")
        
        trades = []
        portfolio_values = [self.initial_capital]
        cash = self.initial_capital
        positions = {ticker: 0 for ticker in data.keys()}
        
        # Nombre de jours à trader
        min_length = min(len(df) for df in data.values())
        
        for day in range(100, min_length - 10):  # Garde 100 jours pour features
            # Extraire données jusqu'à ce jour
            day_data = {ticker: df.iloc[:day+1] for ticker, df in data.items()}
            
            # Créer environnement pour prédiction PPO
            env = UniversalTradingEnvV2(
                data=day_data,
                initial_balance=cash,
                commission=self.commission,
                max_steps=1
            )
            
            obs, _ = env.reset()
            actions, _ = model.predict(obs, deterministic=True)
            
            # Mapper actions
            action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            
            for i, ticker in enumerate(env.tickers):
                action = action_map[int(actions[i])]
                current_price = day_data[ticker]['Close'].iloc[-1]
                
                # Filtrer avec V7 si activé
                if use_v7_filter and v7_predictor and action == 'BUY':
                    v7_result = v7_predictor.predict(ticker, period="3mo")
                    
                    if "error" not in v7_result:
                        if v7_result['prediction'] == 'DOWN' and v7_result['confidence'] > 0.65:
                            action = 'HOLD'  # V7 filtre le BUY
                
                # Exécuter trade
                if action == 'BUY' and positions[ticker] == 0 and cash > 0:
                    qty = int((cash * 0.2) / current_price)  # 20% du capital
                    
                    if qty > 0:
                        cost = qty * current_price * (1 + self.commission + self.slippage)
                        
                        if cost <= cash:
                            cash -= cost
                            positions[ticker] = qty
                            
                            trades.append({
                                'day': day,
                                'ticker': ticker,
                                'action': 'BUY',
                                'qty': qty,
                                'price': current_price,
                                'cost': cost
                            })
                
                elif action == 'SELL' and positions[ticker] > 0:
                    qty = positions[ticker]
                    proceeds = qty * current_price * (1 - self.commission - self.slippage)
                    
                    cash += proceeds
                    positions[ticker] = 0
                    
                    trades.append({
                        'day': day,
                        'ticker': ticker,
                        'action': 'SELL',
                        'qty': qty,
                        'price': current_price,
                        'proceeds': proceeds
                    })
            
            # Calculer valeur portfolio
            positions_value = sum(
                positions[ticker] * day_data[ticker]['Close'].iloc[-1]
                for ticker in positions.keys()
            )
            
            portfolio_values.append(cash + positions_value)
        
        print(f"✅ Simulation terminée: {len(trades)} trades")
        return trades, portfolio_values
    
    def _simulate_v7_trading(self,
                            v7_predictor,
                            data: Dict[str, pd.DataFrame]) -> Tuple[List[Dict], List[float]]:
        """
        Simule le trading avec V7 uniquement
        """
        print(f"\n🔄 Simulation trading V7...")
        
        trades = []
        portfolio_values = [self.initial_capital]
        cash = self.initial_capital
        positions = {ticker: 0 for ticker in data.keys()}
        
        min_length = min(len(df) for df in data.values())
        
        for day in range(100, min_length - 10, 5):  # Check tous les 5 jours
            for ticker in data.keys():
                v7_result = v7_predictor.predict(ticker, period="3mo")
                
                if "error" in v7_result:
                    continue
                
                current_price = data[ticker]['Close'].iloc[day]
                
                # BUY si V7 prédit UP avec confiance > 65%
                if (v7_result['prediction'] == 'UP' and 
                    v7_result['confidence'] > 0.65 and 
                    positions[ticker] == 0 and 
                    cash > 0):
                    
                    qty = int((cash * 0.2) / current_price)
                    
                    if qty > 0:
                        cost = qty * current_price * (1 + self.commission + self.slippage)
                        
                        if cost <= cash:
                            cash -= cost
                            positions[ticker] = qty
                            
                            trades.append({
                                'day': day,
                                'ticker': ticker,
                                'action': 'BUY',
                                'qty': qty,
                                'price': current_price,
                                'cost': cost
                            })
                
                # SELL si V7 prédit DOWN avec confiance > 65%
                elif (v7_result['prediction'] == 'DOWN' and 
                      v7_result['confidence'] > 0.65 and 
                      positions[ticker] > 0):
                    
                    qty = positions[ticker]
                    proceeds = qty * current_price * (1 - self.commission - self.slippage)
                    
                    cash += proceeds
                    positions[ticker] = 0
                    
                    trades.append({
                        'day': day,
                        'ticker': ticker,
                        'action': 'SELL',
                        'qty': qty,
                        'price': current_price,
                        'proceeds': proceeds
                    })
            
            # Portfolio value
            positions_value = sum(
                positions[ticker] * data[ticker]['Close'].iloc[day]
                for ticker in positions.keys()
            )
            
            portfolio_values.append(cash + positions_value)
        
        print(f"✅ Simulation V7 terminée: {len(trades)} trades")
        return trades, portfolio_values
    
    def _calculate_metrics(self, trades: List[Dict], 
                          portfolio_values: List[float]) -> Dict:
        """
        Calcule les métriques de performance
        """
        if not trades or len(portfolio_values) < 2:
            return {
                'total_return': 0,
                'total_trades': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'final_value': self.initial_capital
            }
        
        # Return total
        final_value = portfolio_values[-1]
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
        
        # Trades
        buys = [t for t in trades if t['action'] == 'BUY']
        sells = [t for t in trades if t['action'] == 'SELL']
        
        # Win rate (approximatif)
        wins = sum(1 for s in sells if s['proceeds'] > s.get('cost', 0))
        win_rate = (wins / len(sells) * 100) if sells else 0
        
        # Returns journaliers
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Sharpe Ratio (annualisé)
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Max Drawdown
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = ((peak - value) / peak) * 100
            if dd > max_dd:
                max_dd = dd
        
        return {
            'total_return': round(total_return, 2),
            'total_trades': len(trades),
            'buy_count': len(buys),
            'sell_count': len(sells),
            'win_rate': round(win_rate, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown': round(max_dd, 2),
            'final_value': round(final_value, 2)
        }
    
    def _print_metrics(self, name: str, metrics: Dict):
        """
        Affiche les métriques
        """
        print(f"\n📊 Métriques {name}:")
        print(f"   Return Total: {metrics['total_return']:.2f}%")
        print(f"   Valeur Finale: ${metrics['final_value']:,.2f}")
        print(f"   Trades: {metrics['total_trades']} ({metrics['buy_count']} BUY, {metrics['sell_count']} SELL)")
        print(f"   Win Rate: {metrics['win_rate']:.2f}%")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.2f}%")
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare les résultats des différents modèles
        """
        print("\n" + "="*70)
        print("🏆 COMPARAISON DES MODÈLES")
        print("="*70)
        
        comparison = []
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparison.append({
                'Model': model_name.replace('_', ' ').title(),
                'Return (%)': metrics['total_return'],
                'Final Value ($)': metrics['final_value'],
                'Trades': metrics['total_trades'],
                'Win Rate (%)': metrics['win_rate'],
                'Sharpe': metrics['sharpe_ratio'],
                'Max DD (%)': metrics['max_drawdown']
            })
        
        df = pd.DataFrame(comparison)
        print("\n", df.to_string(index=False))
        
        # Meilleur modèle
        best_model = df.loc[df['Return (%)'].idxmax()]
        print(f"\n🥇 Meilleur modèle: {best_model['Model']}")
        print(f"   Return: {best_model['Return (%)']}%")
        print(f"   Sharpe: {best_model['Sharpe']}")
        
        return df
    
    def save_results(self, output_dir: str = 'tests/backtest_results'):
        """
        Sauvegarde les résultats
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Sauvegarder JSON
        json_file = f"{output_dir}/backtest_{timestamp}.json"
        
        json_results = {}
        for model_name, result in self.results.items():
            json_results[model_name] = {
                'metrics': result['metrics'],
                'trade_count': len(result['trades'])
            }
        
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Résultats sauvegardés: {json_file}")
        
        return json_file


def main():
    """
    Exemple d'utilisation du framework
    """
    print("\n" + "="*70)
    print("🧪 PLOUTOS BACKTESTING FRAMEWORK")
    print("="*70)
    
    # Configuration
    tickers = ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'SPY', 'QQQ']
    start_date = '2023-01-01'
    end_date = '2024-12-01'
    
    # Initialiser framework
    framework = BacktestFramework(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005
    )
    
    # Charger données
    data = framework.load_historical_data(tickers, start_date, end_date)
    
    if len(data) < 3:
        print("❌ Pas assez de données pour le backtest")
        return
    
    # Backtest PPO only
    framework.backtest_ppo_only(
        model_path='models/autonomous/production.zip',
        data=data,
        test_split=0.3
    )
    
    # Backtest V7 only
    if V7_AVAILABLE:
        framework.backtest_v7_only(data=data, test_split=0.3)
    
    # Backtest PPO + V7
    framework.backtest_ppo_plus_v7(
        model_path='models/autonomous/production.zip',
        data=data,
        test_split=0.3
    )
    
    # Comparer
    comparison_df = framework.compare_models()
    
    # Sauvegarder
    framework.save_results()
    
    print("\n" + "="*70)
    print("✅ BACKTEST TERMINÉ")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
