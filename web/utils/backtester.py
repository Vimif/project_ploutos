#!/usr/bin/env python3
"""
📊 PLOUTOS BACKTESTING ENGINE

Moteur de backtesting pour valider stratégies de trading
Calcule: Sharpe Ratio, Max Drawdown, Win Rate, Total Return

Usage:
    backtester = Backtester()
    results = backtester.run(
        ticker='AAPL',
        strategy='rsi_mean_reversion',
        period='1y',
        params={'rsi_oversold': 30, 'rsi_overbought': 70}
    )
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import logging

try:
    import ta
    TA_AVAILABLE = True
except:
    TA_AVAILABLE = False

logger = logging.getLogger(__name__)


class Backtester:
    """
    Moteur de backtesting flexible
    """
    
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001):
        """
        Args:
            initial_capital: Capital initial en $
            commission: Commission par trade (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        logger.info(f"📊 Backtester initialisé (Capital: ${initial_capital:,.0f}, Commission: {commission*100}%)")
    
    def run(self, ticker: str, strategy: str, period: str = '1y', 
            params: Optional[Dict] = None) -> Dict:
        """
        Exécute un backtest
        
        Stratégies disponibles:
            - rsi_mean_reversion: Acheter RSI < threshold, vendre RSI > threshold
            - sma_crossover: Golden Cross / Death Cross (SMA50/SMA200)
            - macd_momentum: Signaux MACD
            - bollinger_bounce: Rebond sur bandes de Bollinger
            - pattern_trading: Trading basé sur patterns (si disponible)
        
        Returns:
            Dict avec résultats: trades, metrics, equity_curve
        """
        params = params or {}
        
        # Télécharger données
        df = yf.download(ticker, period=period, progress=False)
        
        if df.empty:
            return {'error': 'Aucune donnée disponible'}
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Sélectionner et exécuter la stratégie
        if strategy == 'rsi_mean_reversion':
            trades = self._strategy_rsi(df, params)
        elif strategy == 'sma_crossover':
            trades = self._strategy_sma_crossover(df, params)
        elif strategy == 'macd_momentum':
            trades = self._strategy_macd(df, params)
        elif strategy == 'bollinger_bounce':
            trades = self._strategy_bollinger(df, params)
        else:
            return {'error': f"Stratégie '{strategy}' inconnue"}
        
        # Calculer métriques
        metrics = self._calculate_metrics(trades, df)
        
        # Générer equity curve
        equity_curve = self._generate_equity_curve(trades)
        
        return {
            'ticker': ticker,
            'strategy': strategy,
            'period': period,
            'params': params,
            'initial_capital': self.initial_capital,
            'trades': trades,
            'metrics': metrics,
            'equity_curve': equity_curve,
            'timestamp': datetime.now().isoformat()
        }
    
    def _strategy_rsi(self, df: pd.DataFrame, params: dict) -> List[Dict]:
        """
        Stratégie RSI Mean Reversion
        """
        if not TA_AVAILABLE:
            return []
        
        oversold = params.get('rsi_oversold', 30)
        overbought = params.get('rsi_overbought', 70)
        
        df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
        
        trades = []
        position = None
        
        for i in range(len(df)):
            if pd.isna(df['rsi'].iloc[i]):
                continue
            
            date = df.index[i]
            price = df['Close'].iloc[i]
            rsi = df['rsi'].iloc[i]
            
            # Signal d'achat
            if position is None and rsi < oversold:
                position = {
                    'type': 'BUY',
                    'entry_date': date,
                    'entry_price': price,
                    'entry_rsi': rsi,
                    'shares': self.initial_capital / price
                }
            
            # Signal de vente
            elif position is not None and rsi > overbought:
                exit_price = price
                profit = (exit_price - position['entry_price']) * position['shares']
                profit_pct = (exit_price / position['entry_price'] - 1) * 100
                
                # Commission
                commission_cost = (position['entry_price'] + exit_price) * position['shares'] * self.commission
                net_profit = profit - commission_cost
                
                trades.append({
                    'entry_date': position['entry_date'].isoformat(),
                    'entry_price': float(position['entry_price']),
                    'entry_rsi': float(position['entry_rsi']),
                    'exit_date': date.isoformat(),
                    'exit_price': float(exit_price),
                    'exit_rsi': float(rsi),
                    'shares': float(position['shares']),
                    'profit': float(profit),
                    'profit_pct': float(profit_pct),
                    'commission': float(commission_cost),
                    'net_profit': float(net_profit),
                    'duration_days': (date - position['entry_date']).days
                })
                
                position = None
        
        return trades
    
    def _strategy_sma_crossover(self, df: pd.DataFrame, params: dict) -> List[Dict]:
        """
        Stratégie Golden Cross / Death Cross
        """
        if not TA_AVAILABLE:
            return []
        
        fast = params.get('sma_fast', 50)
        slow = params.get('sma_slow', 200)
        
        df['sma_fast'] = ta.trend.sma_indicator(df['Close'], window=fast)
        df['sma_slow'] = ta.trend.sma_indicator(df['Close'], window=slow)
        
        trades = []
        position = None
        
        for i in range(1, len(df)):
            if pd.isna(df['sma_fast'].iloc[i]) or pd.isna(df['sma_slow'].iloc[i]):
                continue
            
            date = df.index[i]
            price = df['Close'].iloc[i]
            
            # Golden Cross (Fast crosses above Slow)
            if position is None:
                if df['sma_fast'].iloc[i-1] <= df['sma_slow'].iloc[i-1] and \
                   df['sma_fast'].iloc[i] > df['sma_slow'].iloc[i]:
                    position = {
                        'entry_date': date,
                        'entry_price': price,
                        'shares': self.initial_capital / price
                    }
            
            # Death Cross (Fast crosses below Slow)
            elif position is not None:
                if df['sma_fast'].iloc[i-1] >= df['sma_slow'].iloc[i-1] and \
                   df['sma_fast'].iloc[i] < df['sma_slow'].iloc[i]:
                    exit_price = price
                    profit = (exit_price - position['entry_price']) * position['shares']
                    profit_pct = (exit_price / position['entry_price'] - 1) * 100
                    commission_cost = (position['entry_price'] + exit_price) * position['shares'] * self.commission
                    
                    trades.append({
                        'entry_date': position['entry_date'].isoformat(),
                        'entry_price': float(position['entry_price']),
                        'exit_date': date.isoformat(),
                        'exit_price': float(exit_price),
                        'shares': float(position['shares']),
                        'profit': float(profit),
                        'profit_pct': float(profit_pct),
                        'commission': float(commission_cost),
                        'net_profit': float(profit - commission_cost),
                        'duration_days': (date - position['entry_date']).days
                    })
                    
                    position = None
        
        return trades
    
    def _strategy_macd(self, df: pd.DataFrame, params: dict) -> List[Dict]:
        """
        Stratégie MACD Momentum
        """
        if not TA_AVAILABLE:
            return []
        
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        trades = []
        position = None
        
        for i in range(1, len(df)):
            if pd.isna(df['macd'].iloc[i]):
                continue
            
            date = df.index[i]
            price = df['Close'].iloc[i]
            
            # MACD crosses above signal
            if position is None:
                if df['macd'].iloc[i-1] <= df['macd_signal'].iloc[i-1] and \
                   df['macd'].iloc[i] > df['macd_signal'].iloc[i]:
                    position = {
                        'entry_date': date,
                        'entry_price': price,
                        'shares': self.initial_capital / price
                    }
            
            # MACD crosses below signal
            elif position is not None:
                if df['macd'].iloc[i-1] >= df['macd_signal'].iloc[i-1] and \
                   df['macd'].iloc[i] < df['macd_signal'].iloc[i]:
                    exit_price = price
                    profit = (exit_price - position['entry_price']) * position['shares']
                    profit_pct = (exit_price / position['entry_price'] - 1) * 100
                    commission_cost = (position['entry_price'] + exit_price) * position['shares'] * self.commission
                    
                    trades.append({
                        'entry_date': position['entry_date'].isoformat(),
                        'entry_price': float(position['entry_price']),
                        'exit_date': date.isoformat(),
                        'exit_price': float(exit_price),
                        'shares': float(position['shares']),
                        'profit': float(profit),
                        'profit_pct': float(profit_pct),
                        'commission': float(commission_cost),
                        'net_profit': float(profit - commission_cost),
                        'duration_days': (date - position['entry_date']).days
                    })
                    
                    position = None
        
        return trades
    
    def _strategy_bollinger(self, df: pd.DataFrame, params: dict) -> List[Dict]:
        """
        Stratégie Bollinger Bands Bounce
        """
        if not TA_AVAILABLE:
            return []
        
        bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_upper'] = bb.bollinger_hband()
        
        trades = []
        position = None
        
        for i in range(len(df)):
            if pd.isna(df['bb_lower'].iloc[i]):
                continue
            
            date = df.index[i]
            price = df['Close'].iloc[i]
            
            # Prix touche bande inférieure
            if position is None and price <= df['bb_lower'].iloc[i]:
                position = {
                    'entry_date': date,
                    'entry_price': price,
                    'shares': self.initial_capital / price
                }
            
            # Prix touche bande supérieure
            elif position is not None and price >= df['bb_upper'].iloc[i]:
                exit_price = price
                profit = (exit_price - position['entry_price']) * position['shares']
                profit_pct = (exit_price / position['entry_price'] - 1) * 100
                commission_cost = (position['entry_price'] + exit_price) * position['shares'] * self.commission
                
                trades.append({
                    'entry_date': position['entry_date'].isoformat(),
                    'entry_price': float(position['entry_price']),
                    'exit_date': date.isoformat(),
                    'exit_price': float(exit_price),
                    'shares': float(position['shares']),
                    'profit': float(profit),
                    'profit_pct': float(profit_pct),
                    'commission': float(commission_cost),
                    'net_profit': float(profit - commission_cost),
                    'duration_days': (date - position['entry_date']).days
                })
                
                position = None
        
        return trades
    
    def _calculate_metrics(self, trades: List[Dict], df: pd.DataFrame) -> Dict:
        """
        Calcule métriques de performance
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'total_return_pct': 0
            }
        
        total_profit = sum(t['net_profit'] for t in trades)
        winning_trades = [t for t in trades if t['net_profit'] > 0]
        losing_trades = [t for t in trades if t['net_profit'] <= 0]
        
        win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0
        
        avg_win = np.mean([t['net_profit'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['net_profit'] for t in losing_trades]) if losing_trades else 0
        
        total_return_pct = (total_profit / self.initial_capital * 100)
        
        # Buy & Hold comparison
        buy_hold_return = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
        
        # Sharpe Ratio (simplifié)
        returns = [t['profit_pct'] for t in trades]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        # Max Drawdown
        equity = [self.initial_capital]
        for t in trades:
            equity.append(equity[-1] + t['net_profit'])
        
        peak = equity[0]
        max_dd = 0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round(win_rate, 2),
            'total_return': round(total_profit, 2),
            'total_return_pct': round(total_return_pct, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown_pct': round(max_dd, 2),
            'buy_hold_return_pct': round(buy_hold_return, 2),
            'vs_buy_hold': round(total_return_pct - buy_hold_return, 2)
        }
    
    def _generate_equity_curve(self, trades: List[Dict]) -> List[Dict]:
        """
        Génère courbe d'équité
        """
        equity = self.initial_capital
        curve = [{'date': 'start', 'equity': equity}]
        
        for t in trades:
            equity += t['net_profit']
            curve.append({
                'date': t['exit_date'],
                'equity': round(equity, 2)
            })
        
        return curve


if __name__ == '__main__':
    # Test
    backtester = Backtester(initial_capital=10000, commission=0.001)
    
    result = backtester.run(
        ticker='AAPL',
        strategy='rsi_mean_reversion',
        period='1y',
        params={'rsi_oversold': 30, 'rsi_overbought': 70}
    )
    
    print("\n📊 BACKTEST RESULTS")
    print("="*50)
    print(f"Ticker: {result['ticker']}")
    print(f"Strategy: {result['strategy']}")
    print(f"Period: {result['period']}")
    print(f"\nMetrics:")
    for key, value in result['metrics'].items():
        print(f"  {key}: {value}")
