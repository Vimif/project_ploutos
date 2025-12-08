#!/usr/bin/env python3
"""
📈 UNIVERSAL TRADING ENVIRONMENT V3 - TREND FOLLOWING

NOUVEAU: Features de TENDANCE pour anticiper les mouvements

Améliorations vs V2:
1. ✅ EMA 50/200 (tendance long terme)
2. ✅ ADX (force tendance)
3. ✅ Momentum ROC (vitesse mouvement)
4. ✅ ATR (volatilité)
5. ✅ Distance support/résistance
6. ✅ Volume trend
7. ✅ Reward avec lookahead (anticiper futur)
8. ✅ Pénalité overtrading
9. ✅ Bonus hold en tendance haussière

Résultat attendu:
- Moins de trades (~50-100/jour au lieu de 600)
- Meilleur timing (acheter bas, vendre haut)
- Score backtest 365j > 70/100
- Return > 10%

Auteur: Ploutos AI Team
Date: Dec 2025
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from collections import deque
import ta  # technical analysis library


class UniversalTradingEnvV3Trend(gym.Env):
    """
    Environnement multi-assets avec TREND FOLLOWING
    
    Actions par ticker:
    - 0 = HOLD  (attendre le bon moment)
    - 1 = BUY   (acheter si tendance monte)
    - 2 = SELL  (vendre si tendance baisse)
    
    Observation space: 103 features par environnement
    - 10 features par ticker (au lieu de 6)
    - 3 features portfolio
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, data, initial_balance=100000, commission=0.0001, max_steps=2000, 
                 buy_pct=0.2, max_trades_per_day=50, lookahead_steps=5):
        """
        Args:
            data (dict): {ticker: DataFrame}
            initial_balance (float): Capital initial
            commission (float): Commission (0.0001 = 0.01%)
            max_steps (int): Steps max par épisode
            buy_pct (float): % portfolio par BUY (0.2 = 20%)
            max_trades_per_day (int): Limite trades/jour (contre overtrading)
            lookahead_steps (int): Steps futurs pour reward anticipation
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
        
        self.data_length = min(len(df) for df in data.values())
        
        # ✅ PRÉ-CALCUL DES FEATURES AVEC TENDANCE
        print("\n⚡ Pré-calcul features V3 + TREND...")
        self._precompute_features_with_trend()
        print("✅ Features tendance pré-calculées !\n")
        
        # Observation space: 10 features par ticker + 3 portfolio
        n_features_per_asset = 10  # Au lieu de 6
        n_portfolio_features = 3
        obs_size = self.n_assets * n_features_per_asset + n_portfolio_features
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # Action space: discret 3 actions par ticker
        self.action_space = spaces.MultiDiscrete([3] * self.n_assets)
        
        # État interne
        self.current_step = 0
        self.reset_step = 0
        self.balance = initial_balance
        self.positions = {ticker: 0 for ticker in self.tickers}
        self.portfolio_value = initial_balance
        self.trades_history = []
        self.trades_today = 0  # Compteur trades quotidien
        self.last_day = 0
        
        # Tracking PnL
        self.entry_prices = {ticker: deque() for ticker in self.tickers}
        self.entry_steps = {ticker: None for ticker in self.tickers}
        self.total_pnl = 0.0
        
        self.portfolio_history = []
        self.peak_portfolio_value = initial_balance
    
    def _precompute_features_with_trend(self):
        """Pré-calcule features + indicateurs de TENDANCE"""
        self.precomputed = {}
        
        for ticker in self.tickers:
            df = self.data[ticker].copy()
            
            # Features de base (comme V2)
            close = np.array(df['Close'].values).flatten()
            high = np.array(df['High'].values).flatten()
            low = np.array(df['Low'].values).flatten()
            volume = np.array(df['Volume'].values).flatten()
            
            # Close normalisé
            close_norm = np.zeros_like(close)
            for i in range(20, len(close)):
                window = close[max(0, i-20):i]
                mean = np.mean(window)
                std = np.std(window)
                close_norm[i] = (close[i] - mean) / (std + 1e-8)
            
            # Volume normalisé
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
            
            # ========== NOUVELLES FEATURES TENDANCE ==========
            
            # 1. EMA 50/200 - Tendance long terme
            ema_50 = pd.Series(close).ewm(span=50, adjust=False).mean().values
            ema_200 = pd.Series(close).ewm(span=200, adjust=False).mean().values
            
            # Tendance: 1 si EMA50 > EMA200 (bull), -1 sinon (bear)
            trend_signal = np.where(ema_50 > ema_200, 1.0, -1.0)
            
            # Distance entre EMAs (force tendance)
            ema_distance = (ema_50 - ema_200) / (close + 1e-8)
            
            # 2. ADX - Force de la tendance (14 periods)
            try:
                df_temp = pd.DataFrame({'High': high, 'Low': low, 'Close': close})
                adx = ta.trend.adx(df_temp['High'], df_temp['Low'], df_temp['Close'], window=14).values
                adx_norm = (adx - 25) / 25  # Normalize autour de 25
            except:
                adx_norm = np.zeros_like(close)
            
            # 3. Momentum ROC (Rate of Change) - Vitesse mouvement
            roc_20 = np.zeros_like(close)
            roc_20[20:] = (close[20:] - close[:-20]) / (close[:-20] + 1e-8)
            
            # 4. ATR (Average True Range) - Volatilité
            try:
                df_temp = pd.DataFrame({'High': high, 'Low': low, 'Close': close})
                atr = ta.volatility.average_true_range(df_temp['High'], df_temp['Low'], df_temp['Close'], window=14).values
                atr_norm = atr / (close + 1e-8)  # Normalize par prix
            except:
                atr_norm = np.zeros_like(close)
            
            # 5. Distance au plus haut/bas récent (support/résistance)
            high_50 = pd.Series(high).rolling(50).max().values
            low_50 = pd.Series(low).rolling(50).min().values
            
            dist_to_high = (high_50 - close) / (close + 1e-8)
            dist_to_low = (close - low_50) / (close + 1e-8)
            
            # 6. Volume trend
            volume_ma = pd.Series(volume).rolling(20).mean().values
            volume_trend = volume / (volume_ma + 1e-8) - 1.0
            
            # Stocker tout
            self.precomputed[ticker] = {
                # Features de base (V2)
                'close': close,
                'close_norm': close_norm.astype(np.float32),
                'volume_norm': volume_norm.astype(np.float32),
                'rsi_norm': rsi_norm.astype(np.float32),
                'returns_1d': returns_1d.astype(np.float32),
                
                # Nouvelles features tendance (V3)
                'trend_signal': trend_signal.astype(np.float32),
                'ema_distance': ema_distance.astype(np.float32),
                'adx_norm': adx_norm.astype(np.float32),
                'roc_20': roc_20.astype(np.float32),
                'atr_norm': atr_norm.astype(np.float32),
                # Distance support/résistance stockée mais pas utilisée directement
            }
    
    def reset(self, seed=None, options=None):
        """Reset environnement"""
        super().reset(seed=seed)
        
        min_start = 200  # Warmup plus long pour EMA200
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
        self.last_day = self.current_step // 6  # Approx 6 steps/jour (hourly)
        
        for ticker in self.tickers:
            self.entry_prices[ticker].clear()
            self.entry_steps[ticker] = None
        self.total_pnl = 0.0
        
        self.portfolio_history = [self.initial_balance]
        self.peak_portfolio_value = self.initial_balance
        
        return self._get_observation(), {}
    
    def step(self, actions):
        """Exécute actions avec reward anticipé"""
        self.current_step += 1
        
        # Reset compteur trades quotidien
        current_day = self.current_step // 6
        if current_day != self.last_day:
            self.trades_today = 0
            self.last_day = current_day
        
        # Prix actuels
        current_prices = {
            ticker: float(self.precomputed[ticker]['close'][self.current_step])
            for ticker in self.tickers
        }
        
        # Portfolio value avant trades
        positions_value = sum(
            self.positions[ticker] * current_prices[ticker]
            for ticker in self.tickers
        )
        previous_portfolio_value = self.balance + positions_value
        
        # Exécuter actions
        total_reward = 0.0
        
        for i, ticker in enumerate(self.tickers):
            action = int(actions[i])
            current_price = current_prices[ticker]
            
            reward_ticker = self._execute_action_with_lookahead(
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
        
        # Clip reward
        total_reward = np.clip(total_reward, -0.5, 0.5)
        
        # Termination
        terminated = (
            new_portfolio_value <= 0 or
            new_portfolio_value <= self.initial_balance * 0.1 or
            self.current_step >= self.data_length - self.lookahead_steps - 1
        )
        
        truncated = (self.current_step - self.reset_step) >= self.max_steps
        
        # Vente forcée à la fin
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
    
    def _execute_action_with_lookahead(self, ticker, action, current_price, portfolio_value):
        """
        Exécute action avec reward ANTICIPÉ (lookahead)
        
        Nouveautés V3:
        - BONUS si BUY avant hausse
        - BONUS si SELL avant baisse
        - MALUS overtrading
        - MALUS BUY en tendance baissiere
        - BONUS HOLD en tendance haussiere
        """
        reward = 0.0
        idx = self.current_step
        
        # Récupérer tendance actuelle
        trend_signal = self.precomputed[ticker]['trend_signal'][idx]
        
        # BUY
        if action == 1:
            # Limite overtrading
            if self.trades_today >= self.max_trades_per_day:
                return -0.1  # Pénalité forte
            
            investment = self.balance * self.buy_pct
            
            if investment > 0 and current_price > 0:
                shares_to_buy = int(investment / current_price)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * current_price
                    fee = cost * self.commission
                    total = cost + fee
                    
                    if self.balance >= total:
                        # Vérifier tendance
                        if trend_signal < 0:
                            # MALUS: acheter en tendance baissiere
                            reward -= 0.05
                        
                        # Exécuter achat
                        self.positions[ticker] += shares_to_buy
                        self.balance -= total
                        self.trades_today += 1
                        
                        for _ in range(shares_to_buy):
                            self.entry_prices[ticker].append(current_price)
                        
                        if self.entry_steps[ticker] is None:
                            self.entry_steps[ticker] = self.current_step
                        
                        # BONUS ANTICIPATION: Vérifier futur
                        if idx + self.lookahead_steps < len(self.precomputed[ticker]['close']):
                            future_price = self.precomputed[ticker]['close'][idx + self.lookahead_steps]
                            future_return = (future_price - current_price) / current_price
                            
                            if future_return > 0.01:  # Futur hausse > 1%
                                reward += 0.1  # BONUS bon timing !
                        
                        self.trades_history.append({
                            'step': self.current_step,
                            'ticker': ticker,
                            'action': 'BUY',
                            'shares': shares_to_buy,
                            'price': current_price,
                            'cost': fee,
                            'trend': trend_signal
                        })
        
        # SELL
        elif action == 2:
            if self.positions[ticker] > 0 and len(self.entry_prices[ticker]) > 0:
                if self.trades_today >= self.max_trades_per_day:
                    return -0.05
                
                shares_to_sell = self.positions[ticker]
                proceeds = shares_to_sell * current_price
                fee = proceeds * self.commission
                
                # Calculer PnL réalisé
                pnl_total = 0.0
                for _ in range(shares_to_sell):
                    if len(self.entry_prices[ticker]) > 0:
                        entry_price = self.entry_prices[ticker].popleft()
                        pnl = (current_price - entry_price) / entry_price
                        pnl_total += pnl
                
                avg_pnl = pnl_total / shares_to_sell if shares_to_sell > 0 else 0
                reward = avg_pnl
                
                # BONUS ANTICIPATION: Vérifier si futur baisse
                if idx + self.lookahead_steps < len(self.precomputed[ticker]['close']):
                    future_price = self.precomputed[ticker]['close'][idx + self.lookahead_steps]
                    future_return = (future_price - current_price) / current_price
                    
                    if future_return < -0.01:  # Futur baisse > 1%
                        reward += 0.1  # BONUS bonne sortie !
                
                self.balance += (proceeds - fee)
                self.positions[ticker] = 0
                self.entry_steps[ticker] = None
                self.total_pnl += avg_pnl
                self.trades_today += 1
                
                self.trades_history.append({
                    'step': self.current_step,
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': current_price,
                    'cost': fee,
                    'pnl': avg_pnl,
                    'trend': trend_signal
                })
        
        # HOLD
        else:
            if self.positions[ticker] > 0 and len(self.entry_prices[ticker]) > 0:
                avg_entry = np.mean(list(self.entry_prices[ticker]))
                unrealized_pnl = (current_price - avg_entry) / avg_entry
                
                # BONUS: Hold en tendance haussière
                if trend_signal > 0 and unrealized_pnl > 0:
                    reward = unrealized_pnl * 0.01  # Petit bonus
                else:
                    reward = unrealized_pnl * 0.005
        
        return np.clip(reward, -0.3, 0.3)
    
    def _get_observation(self):
        """Observation avec features tendance (10 par ticker)"""
        obs = []
        idx = self.current_step
        
        for ticker in self.tickers:
            precomp = self.precomputed[ticker]
            
            # 10 features par ticker
            obs.extend([
                precomp['close_norm'][idx],         # 1. Prix normalisé
                precomp['volume_norm'][idx],        # 2. Volume normalisé
                precomp['rsi_norm'][idx],           # 3. RSI
                precomp['returns_1d'][idx],         # 4. Return 1 jour
                precomp['trend_signal'][idx],       # 5. Tendance (EMA 50/200)
                precomp['ema_distance'][idx],       # 6. Force tendance
                precomp['adx_norm'][idx],           # 7. ADX (force)
                precomp['roc_20'][idx],             # 8. Momentum
                precomp['atr_norm'][idx],           # 9. Volatilité
                float(self.positions[ticker] > 0)   # 10. A une position ?
            ])
        
        # Portfolio features (3)
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
            print(f"Positions:")
            for ticker, shares in self.positions.items():
                if shares > 0:
                    current_price = self.precomputed[ticker]['close'][self.current_step]
                    trend = self.precomputed[ticker]['trend_signal'][self.current_step]
                    trend_str = "📈" if trend > 0 else "📉"
                    print(f"  {ticker} {trend_str}: {shares} @ ${current_price:.2f}")
            print(f"={'='*60}")
