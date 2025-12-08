#!/usr/bin/env python3
"""
🚀 UNIVERSAL TRADING ENVIRONMENT V3 ULTIMATE

VERSION FINALE avec TOUTES les features critiques

Nouvelles capacités ULTIMATE:
1. ✅ Stop-loss dynamique (-5% auto-sell)
2. ✅ Trailing stop (protection gains)
3. ✅ Take-profit automatique (+15%)
4. ✅ Sentiment marché (SPY + VIX)
5. ✅ Position sizing intelligent (ATR)
6. ✅ Multi-timeframe (weekly)
7. ✅ EMA 50/200 (tendance)
8. ✅ ADX (force tendance)
9. ✅ Momentum ROC
10. ✅ Lookahead anticipation
11. ✅ Limite overtrading

Observation: 107 features (vs 63 V2)
Target 365j: Score >75, Return >15%, Drawdown <10%

Auteur: Ploutos AI Team
Date: 8 Dec 2025
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from collections import deque
import ta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


class UniversalTradingEnvV3Trend(gym.Env):
    """
    Environnement ULTIMATE avec:
    - Trend following (EMA, ADX, ROC)
    - Risk management (stop-loss, take-profit)
    - Market sentiment (SPY, VIX)
    - Smart position sizing
    
    Observation: 107 features
    - 11 features/ticker (au lieu de 10)
    - 2 features marché (SPY, VIX)
    - 3 features portfolio
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, data, initial_balance=100000, commission=0.0001, max_steps=2000, 
                 buy_pct=0.2, max_trades_per_day=50, lookahead_steps=5,
                 stop_loss_pct=0.05, trailing_stop=True, take_profit_pct=0.15,
                 use_smart_sizing=True):
        """
        Args:
            data (dict): {ticker: DataFrame}
            initial_balance (float): Capital initial
            commission (float): Commission (0.0001 = 0.01%)
            max_steps (int): Steps max par épisode
            buy_pct (float): % portfolio par BUY (0.2 = 20%)
            max_trades_per_day (int): Limite trades/jour
            lookahead_steps (int): Steps futurs pour anticipation
            stop_loss_pct (float): Stop-loss en % (0.05 = -5%)
            trailing_stop (bool): Activer trailing stop
            take_profit_pct (float): Take-profit en % (0.15 = +15%)
            use_smart_sizing (bool): Position sizing intelligent (ATR)
        """
        super().__init__()
        
        self.data = data
        self.tickers = list(data.keys())
        self.n_assets = len(self.tickers)
        self.initial_balance = initial_balance
        self.commission = commission
        self.max_steps = max_steps
        self.buy_pct = buy_pct
        self.max_trades_per_day = max_trades_per_day
        self.lookahead_steps = lookahead_steps
        
        # ✨ ULTIMATE: Risk management
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop = trailing_stop
        self.take_profit_pct = take_profit_pct
        self.use_smart_sizing = use_smart_sizing
        
        self.data_length = min(len(df) for df in data.values())
        
        # ✅ PRÉ-CALCUL FEATURES ULTIMATE
        print("\n⚡ Pré-calcul features V3 ULTIMATE...")
        self._precompute_all_features()
        print("✅ Features ULTIMATE pré-calculées !\n")
        
        # Observation: 11 features/ticker + 2 marché + 3 portfolio = 107
        n_features_per_asset = 11
        n_market_features = 2
        n_portfolio_features = 3
        obs_size = self.n_assets * n_features_per_asset + n_market_features + n_portfolio_features
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        self.action_space = spaces.MultiDiscrete([3] * self.n_assets)
        
        # État interne
        self.current_step = 0
        self.reset_step = 0
        self.balance = initial_balance
        self.positions = {ticker: 0 for ticker in self.tickers}
        self.portfolio_value = initial_balance
        self.trades_history = []
        self.trades_today = 0
        self.last_day = 0
        
        # Tracking PnL
        self.entry_prices = {ticker: deque() for ticker in self.tickers}
        self.entry_steps = {ticker: None for ticker in self.tickers}
        self.total_pnl = 0.0
        
        # ✨ ULTIMATE: Tracking peaks pour trailing stop
        self.peak_prices = {ticker: None for ticker in self.tickers}
        
        self.portfolio_history = []
        self.peak_portfolio_value = initial_balance
    
    def _precompute_all_features(self):
        """Pré-calcule TOUTES les features (trend + market + multi-TF)"""
        self.precomputed = {}
        
        # ✨ 1. TÉLÉCHARGER SPY ET VIX (marché)
        print("   📊 Téléchargement SPY et VIX...")
        try:
            # Déterminer période des données
            first_ticker = self.tickers[0]
            start_date = self.data[first_ticker].index[0]
            end_date = self.data[first_ticker].index[-1]
            
            spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)
            vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
            
            if not spy_data.empty and not vix_data.empty:
                # SPY trend (bull/bear)
                spy_close = spy_data['Close'].values.flatten()
                spy_ma50 = pd.Series(spy_close).rolling(50).mean().values
                spy_trend = np.where(spy_close > spy_ma50, 1.0, -1.0)
                
                # VIX level (fear/greed)
                vix_close = vix_data['Close'].values.flatten()
                vix_norm = (vix_close - 15) / 15  # Normalize (15 = normal)
                
                self.market_features = {
                    'spy_trend': spy_trend,
                    'vix_norm': vix_norm,
                    'length': len(spy_trend)
                }
                print("      ✅ SPY + VIX chargés")
            else:
                raise Exception("Données vides")
        except Exception as e:
            print(f"      ⚠️  Erreur SPY/VIX: {e}, utilisation valeurs neutres")
            default_len = self.data_length
            self.market_features = {
                'spy_trend': np.zeros(default_len),
                'vix_norm': np.zeros(default_len),
                'length': default_len
            }
        
        # 2. FEATURES PAR TICKER
        for ticker in self.tickers:
            df = self.data[ticker].copy()
            
            close = np.array(df['Close'].values).flatten()
            high = np.array(df['High'].values).flatten()
            low = np.array(df['Low'].values).flatten()
            volume = np.array(df['Volume'].values).flatten()
            
            # Close norm
            close_norm = np.zeros_like(close)
            for i in range(20, len(close)):
                window = close[max(0, i-20):i]
                mean = np.mean(window)
                std = np.std(window)
                close_norm[i] = (close[i] - mean) / (std + 1e-8)
            
            # Volume norm
            volume_norm = np.zeros_like(volume, dtype=np.float32)
            for i in range(20, len(volume)):
                window = volume[max(0, i-20):i]
                mean = np.mean(window)
                std = np.std(window)
                volume_norm[i] = (volume[i] - mean) / (std + 1e-8)
            
            # Returns
            returns_1d = np.zeros_like(close)
            returns_1d[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-8)
            
            # RSI
            rsi = np.zeros_like(close)
            price_diffs = np.zeros(len(close))
            price_diffs[1:] = close[1:] - close[:-1]
            gains = np.where(price_diffs > 0, price_diffs, 0)
            losses = np.where(price_diffs < 0, -price_diffs, 0)
            
            for i in range(14, len(close)):
                avg_gain = np.mean(gains[max(0, i-14):i])
                avg_loss = np.mean(losses[max(0, i-14):i])
                if avg_loss == 0:
                    rsi[i] = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi[i] = 100 - (100 / (1 + rs))
            
            rsi_norm = (rsi - 50) / 50
            
            # EMA 50/200 - Tendance daily
            ema_50 = pd.Series(close).ewm(span=50, adjust=False).mean().values
            ema_200 = pd.Series(close).ewm(span=200, adjust=False).mean().values
            trend_signal = np.where(ema_50 > ema_200, 1.0, -1.0)
            ema_distance = (ema_50 - ema_200) / (close + 1e-8)
            
            # ✨ ULTIMATE: Tendance WEEKLY (multi-timeframe)
            try:
                df_weekly = df.resample('W').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                })
                ema_10w = df_weekly['Close'].ewm(span=10).mean()
                ema_40w = df_weekly['Close'].ewm(span=40).mean()
                trend_weekly = (ema_10w > ema_40w).astype(float)
                trend_weekly_daily = trend_weekly.reindex(df.index, method='ffill').values
            except:
                trend_weekly_daily = trend_signal  # Fallback
            
            # ADX
            try:
                df_temp = pd.DataFrame({'High': high, 'Low': low, 'Close': close})
                adx = ta.trend.adx(df_temp['High'], df_temp['Low'], df_temp['Close'], window=14).values
                adx_norm = (adx - 25) / 25
            except:
                adx_norm = np.zeros_like(close)
            
            # Momentum ROC
            roc_20 = np.zeros_like(close)
            roc_20[20:] = (close[20:] - close[:-20]) / (close[:-20] + 1e-8)
            
            # ATR (pour position sizing)
            try:
                df_temp = pd.DataFrame({'High': high, 'Low': low, 'Close': close})
                atr = ta.volatility.average_true_range(df_temp['High'], df_temp['Low'], df_temp['Close'], window=14).values
                atr_norm = atr / (close + 1e-8)
            except:
                atr_norm = np.ones_like(close) * 0.02  # Default 2%
            
            self.precomputed[ticker] = {
                'close': close,
                'close_norm': close_norm.astype(np.float32),
                'volume_norm': volume_norm.astype(np.float32),
                'rsi_norm': rsi_norm.astype(np.float32),
                'returns_1d': returns_1d.astype(np.float32),
                'trend_signal': trend_signal.astype(np.float32),
                'ema_distance': ema_distance.astype(np.float32),
                'adx_norm': adx_norm.astype(np.float32),
                'roc_20': roc_20.astype(np.float32),
                'atr_norm': atr_norm.astype(np.float32),
                'trend_weekly': trend_weekly_daily.astype(np.float32),  # ✨ NEW
            }
    
    def reset(self, seed=None, options=None):
        """Reset environnement"""
        super().reset(seed=seed)
        
        min_start = 200
        max_end = self.data_length - self.max_steps - self.lookahead_steps
        
        if max_end <= min_start:
            self.current_step = min(min_start, self.data_length - self.max_steps - self.lookahead_steps - 1)
            if self.current_step < 0:
                self.current_step = 0
        else:
            self.current_step = np.random.randint(min_start, max_end)
        
        self.reset_step = self.current_step
        self.balance = self.initial_balance
        self.positions = {ticker: 0 for ticker in self.tickers}
        self.portfolio_value = self.initial_balance
        self.trades_history = []
        self.trades_today = 0
        self.last_day = self.current_step // 6
        
        for ticker in self.tickers:
            self.entry_prices[ticker].clear()
            self.entry_steps[ticker] = None
            self.peak_prices[ticker] = None  # ✨ Reset trailing
        self.total_pnl = 0.0
        
        self.portfolio_history = [self.initial_balance]
        self.peak_portfolio_value = self.initial_balance
        
        return self._get_observation(), {}
    
    def step(self, actions):
        """Exécute actions avec risk management ULTIMATE"""
        self.current_step += 1
        
        current_day = self.current_step // 6
        if current_day != self.last_day:
            self.trades_today = 0
            self.last_day = current_day
        
        current_prices = {
            ticker: float(self.precomputed[ticker]['close'][self.current_step])
            for ticker in self.tickers
        }
        
        positions_value = sum(
            self.positions[ticker] * current_prices[ticker]
            for ticker in self.tickers
        )
        previous_portfolio_value = self.balance + positions_value
        
        # ✨ ULTIMATE: VÉRIFIER STOP-LOSS / TAKE-PROFIT AVANT ACTIONS
        total_reward = 0.0
        forced_actions = {}  # Actions forcées par risk management
        
        for ticker in self.tickers:
            if self.positions[ticker] > 0 and len(self.entry_prices[ticker]) > 0:
                current_price = current_prices[ticker]
                avg_entry = np.mean(list(self.entry_prices[ticker]))
                pnl_pct = (current_price - avg_entry) / avg_entry
                
                # STOP-LOSS: Vente forcée si perte > seuil
                if pnl_pct < -self.stop_loss_pct:
                    forced_actions[ticker] = 2  # SELL
                    total_reward -= 0.15  # Pénalité perte
                    continue
                
                # TAKE-PROFIT: Vente forcée si gain > seuil
                if pnl_pct > self.take_profit_pct:
                    forced_actions[ticker] = 2  # SELL
                    total_reward += 0.05  # Bonus prise bénéfice
                    continue
                
                # TRAILING STOP
                if self.trailing_stop:
                    if self.peak_prices[ticker] is None:
                        self.peak_prices[ticker] = current_price
                    else:
                        self.peak_prices[ticker] = max(self.peak_prices[ticker], current_price)
                    
                    drawdown_from_peak = (current_price - self.peak_prices[ticker]) / self.peak_prices[ticker]
                    if drawdown_from_peak < -self.stop_loss_pct:
                        forced_actions[ticker] = 2  # SELL
                        total_reward += 0.02  # Petit bonus protection
        
        # Exécuter actions (forcées ou choisies)
        for i, ticker in enumerate(self.tickers):
            if ticker in forced_actions:
                action = forced_actions[ticker]
            else:
                action = int(actions[i])
            
            current_price = current_prices[ticker]
            
            reward_ticker = self._execute_action_ultimate(
                ticker, action, current_price, previous_portfolio_value
            )
            total_reward += reward_ticker
        
        # Recalculer portfolio
        positions_value = sum(
            self.positions[ticker] * current_prices[ticker]
            for ticker in self.tickers
        )
        new_portfolio_value = self.balance + positions_value
        self.portfolio_value = new_portfolio_value
        
        self.portfolio_history.append(new_portfolio_value)
        self.peak_portfolio_value = max(self.peak_portfolio_value, new_portfolio_value)
        
        total_reward = np.clip(total_reward, -0.5, 0.5)
        
        terminated = (
            new_portfolio_value <= 0 or
            new_portfolio_value <= self.initial_balance * 0.1 or
            self.current_step >= self.data_length - self.lookahead_steps - 1
        )
        
        truncated = (self.current_step - self.reset_step) >= self.max_steps
        
        if (terminated or truncated):
            for ticker in self.tickers:
                if self.positions[ticker] > 0 and len(self.entry_prices[ticker]) > 0:
                    current_price = current_prices[ticker]
                    avg_entry = np.mean(list(self.entry_prices[ticker]))
                    final_pnl = (current_price - avg_entry) / avg_entry
                    total_reward += final_pnl
                    
                    proceeds = self.positions[ticker] * current_price * (1 - self.commission)
                    self.balance += proceeds
                    self.positions[ticker] = 0
                    self.portfolio_value = self.balance
        
        info = {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'positions': self.positions.copy(),
            'n_trades': len(self.trades_history),
            'trades_today': self.trades_today,
            'peak_value': self.peak_portfolio_value,
            'total_pnl': self.total_pnl
        }
        
        return self._get_observation(), total_reward, terminated, truncated, info
    
    def _execute_action_ultimate(self, ticker, action, current_price, portfolio_value):
        """Exécute action avec TOUTES les features ULTIMATE"""
        reward = 0.0
        idx = self.current_step
        
        trend_daily = self.precomputed[ticker]['trend_signal'][idx]
        trend_weekly = self.precomputed[ticker]['trend_weekly'][idx]
        atr = self.precomputed[ticker]['atr_norm'][idx]
        adx = self.precomputed[ticker]['adx_norm'][idx]
        
        # Marché
        spy_idx = min(idx, self.market_features['length'] - 1)
        spy_trend = self.market_features['spy_trend'][spy_idx]
        vix_level = self.market_features['vix_norm'][spy_idx]
        
        # BUY
        if action == 1:
            if self.trades_today >= self.max_trades_per_day:
                return -0.1
            
            # ✨ ULTIMATE: Position sizing intelligent
            if self.use_smart_sizing:
                # Volatility factor (ATR)
                volatility_factor = 1.0 / (1.0 + atr * 5)
                # Confidence factor (ADX)
                confidence_factor = min((adx + 1.0) / 2.0, 1.0)
                position_pct = self.buy_pct * volatility_factor * confidence_factor
                position_pct = np.clip(position_pct, 0.05, 0.30)
            else:
                position_pct = self.buy_pct
            
            investment = self.balance * position_pct
            
            if investment > 0 and current_price > 0:
                shares_to_buy = int(investment / current_price)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    fee = cost * self.commission
                    total = cost + fee
                    
                    if self.balance >= total:
                        # MALUS: Acheter contre tendance
                        if trend_daily < 0 or trend_weekly < 0:
                            reward -= 0.05
                        
                        # ✨ ULTIMATE: MALUS marché baissier
                        if spy_trend < 0:
                            reward -= 0.03
                        
                        # ✨ ULTIMATE: MALUS VIX élevé (panique)
                        if vix_level > 1.0:  # VIX > 30
                            reward -= 0.02
                        
                        # Exécuter
                        self.positions[ticker] += shares_to_buy
                        self.balance -= total
                        self.trades_today += 1
                        self.peak_prices[ticker] = current_price  # Init trailing
                        
                        for _ in range(shares_to_buy):
                            self.entry_prices[ticker].append(current_price)
                        
                        if self.entry_steps[ticker] is None:
                            self.entry_steps[ticker] = self.current_step
                        
                        # Lookahead
                        if idx + self.lookahead_steps < len(self.precomputed[ticker]['close']):
                            future_price = self.precomputed[ticker]['close'][idx + self.lookahead_steps]
                            future_return = (future_price - current_price) / current_price
                            
                            if future_return > 0.01:
                                reward += 0.1  # BONUS anticipation
                        
                        self.trades_history.append({
                            'step': self.current_step,
                            'ticker': ticker,
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': current_price,
                            'cost': fee,
                            'trend_daily': trend_daily,
                            'trend_weekly': trend_weekly,
                            'spy_trend': spy_trend
                        })
        
        # SELL
        elif action == 2:
            if self.positions[ticker] > 0 and len(self.entry_prices[ticker]) > 0:
                if self.trades_today >= self.max_trades_per_day:
                    return -0.05
                
                shares_to_sell = self.positions[ticker]
                proceeds = shares_to_sell * current_price
                fee = proceeds * self.commission
                
                pnl_total = 0.0
                for _ in range(shares_to_sell):
                    if len(self.entry_prices[ticker]) > 0:
                        entry_price = self.entry_prices[ticker].popleft()
                        pnl = (current_price - entry_price) / entry_price
                        pnl_total += pnl
                
                avg_pnl = pnl_total / shares_to_sell if shares_to_sell > 0 else 0
                reward = avg_pnl
                
                # Lookahead
                if idx + self.lookahead_steps < len(self.precomputed[ticker]['close']):
                    future_price = self.precomputed[ticker]['close'][idx + self.lookahead_steps]
                    future_return = (future_price - current_price) / current_price
                    
                    if future_return < -0.01:
                        reward += 0.1  # BONUS sortie avant baisse
                
                self.balance += (proceeds - fee)
                self.positions[ticker] = 0
                self.entry_steps[ticker] = None
                self.peak_prices[ticker] = None  # Reset trailing
                self.total_pnl += avg_pnl
                self.trades_today += 1
                
                self.trades_history.append({
                    'step': self.current_step,
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'cost': fee,
                    'pnl': avg_pnl
                })
        
        # HOLD
        else:
            if self.positions[ticker] > 0 and len(self.entry_prices[ticker]) > 0:
                avg_entry = np.mean(list(self.entry_prices[ticker]))
                unrealized_pnl = (current_price - avg_entry) / avg_entry
                
                # BONUS: Hold en double tendance haussière
                if trend_daily > 0 and trend_weekly > 0 and unrealized_pnl > 0:
                    reward = unrealized_pnl * 0.015  # Bonus renforcé
                elif trend_daily > 0 and unrealized_pnl > 0:
                    reward = unrealized_pnl * 0.01
                else:
                    reward = unrealized_pnl * 0.005
            
            # ✨ ULTIMATE: Bonus cash si marché en panique
            elif vix_level > 1.5 and self.balance > self.initial_balance * 0.5:
                reward += 0.01  # Bonus protection
        
        return np.clip(reward, -0.3, 0.3)
    
    def _get_observation(self):
        """Observation ULTIMATE: 107 features"""
        obs = []
        idx = self.current_step
        
        # 11 features par ticker
        for ticker in self.tickers:
            precomp = self.precomputed[ticker]
            
            obs.extend([
                precomp['close_norm'][idx],
                precomp['volume_norm'][idx],
                precomp['rsi_norm'][idx],
                precomp['returns_1d'][idx],
                precomp['trend_signal'][idx],      # Daily
                precomp['trend_weekly'][idx],       # ✨ Weekly
                precomp['ema_distance'][idx],
                precomp['adx_norm'][idx],
                precomp['roc_20'][idx],
                precomp['atr_norm'][idx],
                float(self.positions[ticker] > 0)
            ])
        
        # ✨ 2 features marché
        spy_idx = min(idx, self.market_features['length'] - 1)
        obs.extend([
            float(self.market_features['spy_trend'][spy_idx]),
            float(self.market_features['vix_norm'][spy_idx])
        ])
        
        # 3 features portfolio
        cash_ratio = self.balance / self.portfolio_value if self.portfolio_value > 0 else 0
        total_value_norm = (self.portfolio_value - self.initial_balance) / self.initial_balance
        n_positions = sum(1 for pos in self.positions.values() if pos > 0) / self.n_assets
        
        obs.extend([
            float(cash_ratio),
            float(total_value_norm),
            float(n_positions)
        ])
        
        obs = np.array(obs, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs
    
    def render(self, mode='human'):
        """Afficher état"""
        if mode == 'human':
            print(f"\n{'='*60}")
            print(f"Step: {self.current_step}")
            print(f"Portfolio: ${self.portfolio_value:,.2f}")
            print(f"Cash: ${self.balance:,.2f}")
            print(f"Trades today: {self.trades_today}/{self.max_trades_per_day}")
            print(f"Total PnL: {self.total_pnl:.4f}")
            
            idx = min(self.current_step, self.market_features['length'] - 1)
            spy_trend = self.market_features['spy_trend'][idx]
            vix = self.market_features['vix_norm'][idx]
            print(f"Marché: SPY {'BULL 📈' if spy_trend > 0 else 'BEAR 📉'}, VIX {vix:.2f}")
            
            print(f"Positions:")
            for ticker, shares in self.positions.items():
                if shares > 0:
                    current_price = self.precomputed[ticker]['close'][self.current_step]
                    avg_entry = np.mean(list(self.entry_prices[ticker]))
                    pnl = (current_price - avg_entry) / avg_entry * 100
                    trend_d = self.precomputed[ticker]['trend_signal'][self.current_step]
                    trend_w = self.precomputed[ticker]['trend_weekly'][self.current_step]
                    trend_str = "📈📈" if (trend_d > 0 and trend_w > 0) else "📈" if trend_d > 0 else "📉"
                    print(f"  {ticker} {trend_str}: {shares} @ ${current_price:.2f} (PnL: {pnl:+.1f}%)")
            print(f"={'='*60}")
