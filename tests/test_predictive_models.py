#!/usr/bin/env python3
"""
🔮 PLOUTOS PREDICTIVE MODELS TESTER

Test et compare les modèles prédictifs (V7, futures variantes)
SANS utiliser PPO

Usage:
    python tests/test_predictive_models.py --days 90 --tickers NVDA,MSFT,AAPL
    python tests/test_predictive_models.py --preset tech
    python tests/test_predictive_models.py --full  # Test complet

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List

try:
    from src.models.v7_predictor import V7Predictor
    V7_AVAILABLE = True
except:
    V7_AVAILABLE = False
    print("❌ V7 Predictor non disponible")
    sys.exit(1)

# Presets de tickers
TICKER_PRESETS = {
    'tech': ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'TSLA'],
    'finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK'],
    'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC'],
    'defensive': ['SPY', 'QQQ', 'VOO', 'VTI', 'IWM', 'DIA', 'VEA'],
    'mixed': ['NVDA', 'MSFT', 'JPM', 'XOM', 'SPY', 'QQQ', 'AAPL'],
    'full': ['NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'JPM', 'SPY', 'QQQ', 'XOM', 'CVX']
}


class PredictiveModelTester:
    """
    Testeur pour modèles prédictifs uniquement
    """
    
    def __init__(self, capital: float = 100000, commission: float = 0.001):
        self.capital = capital
        self.commission = commission
        self.results = {}
    
    def load_data(self, tickers: List[str], days: int) -> Dict[str, pd.DataFrame]:
        """
        Charge les données historiques
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 100)  # +100 pour features
        
        print(f"\n📡 Chargement données...")
        print(f"   Période: {start_date.date()} à {end_date.date()}")
        print(f"   Tickers: {', '.join(tickers)}")
        
        data = {}
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                if not df.empty and len(df) > 50:
                    data[ticker] = df
                    print(f"   ✅ {ticker}: {len(df)} jours")
                else:
                    print(f"   ⚠️  {ticker}: Données insuffisantes")
            except Exception as e:
                print(f"   ❌ {ticker}: {e}")
        
        print(f"\n✅ {len(data)}/{len(tickers)} tickers chargés\n")
        return data
    
    def test_v7_momentum(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Test V7 Enhanced Momentum (68.35% accuracy)
        """
        print("\n" + "="*70)
        print("⭐ TEST: V7 ENHANCED MOMENTUM (68.35% accuracy)")
        print("="*70)
        
        v7 = V7Predictor()
        if not v7.load("momentum"):
            print("❌ Erreur chargement V7")
            return {}
        
        print("✅ Modèle chargé")
        
        # Tester chaque ticker
        predictions = {}
        correct = 0
        total = 0
        
        for ticker, df in data.items():
            print(f"\n🔍 Test {ticker}...")
            
            # Prendre 70% pour historique, 30% pour test
            split_idx = int(len(df) * 0.7)
            test_days = list(range(split_idx, len(df) - 10, 10))  # Check tous les 10 jours
            
            ticker_preds = []
            
            for day in test_days:
                # Prédire
                result = v7.predict(ticker, period="3mo")
                
                if "error" in result:
                    continue
                
                # Vérifier avec les 10 jours suivants
                current_price = df['Close'].iloc[day]
                future_price = df['Close'].iloc[min(day + 10, len(df) - 1)]
                
                actual_direction = 'UP' if future_price > current_price else 'DOWN'
                predicted = result['prediction']
                confidence = result['confidence']
                
                is_correct = (predicted == actual_direction)
                
                if is_correct:
                    correct += 1
                total += 1
                
                ticker_preds.append({
                    'day': day,
                    'predicted': predicted,
                    'actual': actual_direction,
                    'confidence': confidence,
                    'correct': is_correct
                })
            
            predictions[ticker] = ticker_preds
            
            # Stats ticker
            ticker_correct = sum(1 for p in ticker_preds if p['correct'])
            ticker_accuracy = (ticker_correct / len(ticker_preds) * 100) if ticker_preds else 0
            print(f"   Accuracy {ticker}: {ticker_accuracy:.2f}% ({ticker_correct}/{len(ticker_preds)})")
        
        # Stats globales
        overall_accuracy = (correct / total * 100) if total > 0 else 0
        
        print(f"\n📊 Résultats Globaux:")
        print(f"   Accuracy: {overall_accuracy:.2f}% ({correct}/{total})")
        print(f"   Tickers testés: {len(predictions)}")
        print(f"   Prédictions totales: {total}")
        
        metrics = {
            'model': 'V7 Enhanced Momentum',
            'overall_accuracy': round(overall_accuracy, 2),
            'correct_predictions': correct,
            'total_predictions': total,
            'tickers_tested': len(predictions),
            'per_ticker': {}
        }
        
        for ticker, preds in predictions.items():
            ticker_correct = sum(1 for p in preds if p['correct'])
            metrics['per_ticker'][ticker] = {
                'accuracy': round(ticker_correct / len(preds) * 100, 2) if preds else 0,
                'correct': ticker_correct,
                'total': len(preds)
            }
        
        self.results['v7_momentum'] = metrics
        return metrics
    
    def test_trading_performance(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Simule des trades avec V7 et calcule le PnL
        """
        print("\n" + "="*70)
        print("💰 TEST: PERFORMANCE TRADING AVEC V7")
        print("="*70)
        
        v7 = V7Predictor()
        if not v7.load("momentum"):
            print("❌ Erreur chargement V7")
            return {}
        
        cash = self.capital
        positions = {ticker: 0 for ticker in data.keys()}
        trades = []
        portfolio_values = [cash]
        
        # Split 70/30
        min_length = min(len(df) for df in data.values())
        split_idx = int(min_length * 0.7)
        
        print(f"\n🔄 Simulation trading...")
        print(f"   Capital initial: ${cash:,.2f}")
        print(f"   Période test: {split_idx} à {min_length} jours")
        
        for day in range(split_idx, min_length - 10, 5):  # Check tous les 5 jours
            
            for ticker in data.keys():
                result = v7.predict(ticker, period="3mo")
                
                if "error" in result:
                    continue
                
                current_price = data[ticker]['Close'].iloc[day]
                
                # BUY si UP avec confiance > 65%
                if (result['prediction'] == 'UP' and 
                    result['confidence'] > 0.65 and 
                    positions[ticker] == 0 and 
                    cash > 1000):
                    
                    # Investir 15% du capital disponible
                    invest_amount = cash * 0.15
                    qty = int(invest_amount / current_price)
                    
                    if qty > 0:
                        cost = qty * current_price * (1 + self.commission)
                        
                        if cost <= cash:
                            cash -= cost
                            positions[ticker] = qty
                            
                            trades.append({
                                'day': day,
                                'ticker': ticker,
                                'action': 'BUY',
                                'qty': qty,
                                'price': current_price,
                                'cost': cost,
                                'confidence': result['confidence']
                            })
                            print(f"   🟢 BUY {ticker} @ ${current_price:.2f} (conf: {result['confidence']:.2%})")
                
                # SELL si DOWN avec confiance > 65% OU si position en perte > 5%
                elif positions[ticker] > 0:
                    should_sell = False
                    reason = ""
                    
                    if result['prediction'] == 'DOWN' and result['confidence'] > 0.65:
                        should_sell = True
                        reason = f"V7 DOWN (conf: {result['confidence']:.2%})"
                    
                    # Stop loss à -5%
                    buy_trade = next((t for t in reversed(trades) 
                                     if t['ticker'] == ticker and t['action'] == 'BUY'), None)
                    
                    if buy_trade:
                        pnl_pct = ((current_price - buy_trade['price']) / buy_trade['price']) * 100
                        if pnl_pct < -5:
                            should_sell = True
                            reason = f"Stop Loss ({pnl_pct:.2f}%)"
                    
                    if should_sell:
                        qty = positions[ticker]
                        proceeds = qty * current_price * (1 - self.commission)
                        
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
                        
                        pnl = proceeds - buy_trade['cost'] if buy_trade else 0
                        print(f"   🔴 SELL {ticker} @ ${current_price:.2f} ({reason}) PnL: ${pnl:.2f}")
            
            # Portfolio value
            positions_value = sum(
                positions[ticker] * data[ticker]['Close'].iloc[day]
                for ticker in positions.keys()
            )
            portfolio_values.append(cash + positions_value)
        
        # Calculer métriques
        final_value = portfolio_values[-1]
        total_return = ((final_value - self.capital) / self.capital) * 100
        
        buys = [t for t in trades if t['action'] == 'BUY']
        sells = [t for t in trades if t['action'] == 'SELL']
        
        # Calculer PnL des trades fermés
        closed_trades_pnl = []
        for sell in sells:
            buy = next((t for t in reversed(buys) 
                       if t['ticker'] == sell['ticker'] and t['day'] < sell['day']), None)
            if buy:
                pnl = sell['proceeds'] - buy['cost']
                closed_trades_pnl.append(pnl)
        
        wins = sum(1 for pnl in closed_trades_pnl if pnl > 0)
        win_rate = (wins / len(closed_trades_pnl) * 100) if closed_trades_pnl else 0
        
        print(f"\n📊 Résultats Trading:")
        print(f"   Return Total: {total_return:.2f}%")
        print(f"   Valeur Finale: ${final_value:,.2f}")
        print(f"   Trades: {len(trades)} ({len(buys)} BUY, {len(sells)} SELL)")
        print(f"   Win Rate: {win_rate:.2f}% ({wins}/{len(closed_trades_pnl)})")
        
        metrics = {
            'model': 'V7 Enhanced Momentum',
            'total_return': round(total_return, 2),
            'final_value': round(final_value, 2),
            'total_trades': len(trades),
            'buy_count': len(buys),
            'sell_count': len(sells),
            'win_rate': round(win_rate, 2),
            'closed_trades': len(closed_trades_pnl)
        }
        
        self.results['v7_trading'] = metrics
        return metrics
    
    def save_results(self):
        """
        Sauvegarde les résultats
        """
        output_dir = Path('tests/predictive_results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_file = output_dir / f"predictive_test_{timestamp}.json"
        
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Résultats sauvegardés: {json_file}")
        return json_file


def main():
    parser = argparse.ArgumentParser(description='Test des modèles prédictifs')
    
    parser.add_argument('--preset', type=str, choices=list(TICKER_PRESETS.keys()),
                       help='Preset de tickers')
    parser.add_argument('--tickers', type=str,
                       help='Liste de tickers (séparés par virgules)')
    parser.add_argument('--days', type=int, default=90,
                       help='Nombre de jours pour le test (défaut: 90)')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Capital initial (défaut: 100000)')
    parser.add_argument('--accuracy-only', action='store_true',
                       help='Tester uniquement la précision (pas de trading)')
    parser.add_argument('--trading-only', action='store_true',
                       help='Tester uniquement le trading (pas d\'accuracy)')
    
    args = parser.parse_args()
    
    # Déterminer tickers
    if args.preset:
        tickers = TICKER_PRESETS[args.preset]
        print(f"✅ Preset '{args.preset}' sélectionné")
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
    else:
        tickers = TICKER_PRESETS['mixed']
        print("ℹ️  Aucun ticker spécifié - utilise preset 'mixed'")
    
    print("\n" + "="*70)
    print("🔮 PLOUTOS PREDICTIVE MODELS TESTER")
    print("="*70)
    print(f"\n🎯 Tickers: {', '.join(tickers)}")
    print(f"📅 Période: {args.days} derniers jours")
    print(f"💰 Capital: ${args.capital:,.0f}")
    
    # Initialiser
    tester = PredictiveModelTester(capital=args.capital)
    
    # Charger données
    data = tester.load_data(tickers, args.days)
    
    if len(data) == 0:
        print("❌ Aucune donnée chargée")
        return
    
    # Tests
    if not args.trading_only:
        print("\n🔵 Test 1: Accuracy...")
        tester.test_v7_momentum(data)
    
    if not args.accuracy_only:
        print("\n🟡 Test 2: Trading Performance...")
        tester.test_trading_performance(data)
    
    # Sauvegarder
    tester.save_results()
    
    print("\n" + "="*70)
    print("✅ TESTS TERMINÉS")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
