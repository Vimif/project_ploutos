#!/usr/bin/env python3
"""
✅ Environnement V6 - MEILLEUR TIMING avec Features V2 et Récompense Avancée

Objectif: Résoudre le problème "buy high" et "never sell" (0% win rate)

Améliorations:
- Features V2 optimisées
- Récompense Differential Sharpe Ratio (DSR)
- Support/Resistance dynamiques
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict
from collections import deque

from ploutos.features.pipeline import FeaturePipeline
from ploutos.env.rewards import AdvancedRewardCalculator

class TradingEnvironment(gym.Env):
    """Environnement avec Features V2 pour meilleur timing"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        initial_balance: float = 100000.0,
        commission: float = 0.0,
        sec_fee: float = 0.0000221,
        finra_taf: float = 0.000145,
        max_steps: int = 2500,
        buy_pct: float = 0.20,
        slippage_model: str = 'realistic',
        spread_bps: float = 2.0,
        market_impact_factor: float = 0.0001,
        max_position_pct: float = 0.25,
        reward_scaling: float = 1.0,  # Réduit car DSR est déjà calibré
        use_sharpe_penalty: bool = True,
        use_drawdown_penalty: bool = True,
        max_trades_per_day: int = 10,
        min_holding_period: int = 2,
        reward_trade_success: float = 0.5,
        penalty_overtrading: float = 0.005,
        drawdown_penalty_factor: float = 3.0,
    ):
        super().__init__()
        
        self.data = data
        self.tickers = list(data.keys())
        self.n_assets = len(self.tickers)
        
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        
        self.commission = commission
        self.sec_fee = sec_fee
        self.finra_taf = finra_taf
        
        self.slippage_model = slippage_model
        self.spread_bps = spread_bps / 10000
        self.market_impact_factor = market_impact_factor
        
        self.max_position_pct = max_position_pct
        self.buy_pct = buy_pct
        
        self.reward_scaling = reward_scaling
        self.max_trades_per_day = max_trades_per_day
        self.min_holding_period = min_holding_period
        
        self.current_step = 0
        self.max_steps = max_steps
        self.done = False
        
        self.portfolio = {ticker: 0.0 for ticker in self.tickers}
        self.entry_prices = {ticker: 0.0 for ticker in self.tickers}
        self.portfolio_value_history = deque(maxlen=252)
        self.returns_history = deque(maxlen=100)
        self.peak_value = initial_balance
        
        self.trades_today = 0
        self.last_trade_step = {ticker: -999 for ticker in self.tickers}
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # ✅ Calculateur de récompense avancé
        self.reward_calculator = AdvancedRewardCalculator()
        
        # ✅ Préparer les features V2
        self._prepare_features_v2()
        
        # Définir les spaces
        n_features_per_ticker = len(self.feature_columns)
        obs_size = self.n_assets * n_features_per_ticker + self.n_assets + 3
        
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        self.action_space = gym.spaces.MultiDiscrete([3] * self.n_assets)
        
        print(f"Env V6 (DSR Rewards): {self.n_assets} tickers x {n_features_per_ticker} features = {obs_size} dims")
    
    def _prepare_features_v2(self):
        """Préparer Features V2 optimisées"""
        print(f"  [Env] Calcul Features V2 (optimisees pour timing)...")
        
        self.processed_data = {}
        self.feature_engineer = FeaturePipeline()
        
        for ticker in self.tickers:
            df = self.data[ticker].copy()
            df = self.feature_engineer.calculate_all_features(df)
            self.processed_data[ticker] = df
        
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.feature_columns = [
            col for col in self.processed_data[self.tickers[0]].columns
            if col not in exclude_cols
        ]
        
        print(f"  ✅ {len(self.feature_columns)} features calculées par ticker")
        
        self.max_steps = min(
            self.max_steps,
            min(len(df) for df in self.processed_data.values()) - 100
        )
    
    def reset(self, seed=None, options=None):
        """Reset l'environnement pour un nouvel épisode"""
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_value = self.initial_balance
        
        self.portfolio = {ticker: 0.0 for ticker in self.tickers}
        self.entry_prices = {ticker: 0.0 for ticker in self.tickers}
        self.portfolio_value_history.clear()
        self.returns_history.clear()
        
        self.reward_calculator.reset()  # Reset DSR
        
        self.current_step = np.random.randint(100, max(101, self.max_steps // 2))
        self.trades_today = 0
        self.last_trade_step = {ticker: -999 for ticker in self.tickers}
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.done = False
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, actions):
        """Un step dans l'environnement"""
        if self.done:
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        # Reset les trades du jour
        if self.current_step % 78 == 0:
            self.trades_today = 0
        
        trades_executed = 0
        winning_trade_this_step = False
        
        # Exécuter les actions
        for i, (ticker, action) in enumerate(zip(self.tickers, actions)):
            trade_res = self._execute_trade(ticker, action, i)
            if action != 0:
                trades_executed += 1
            if trade_res > 0.05: # Si trade gagnant
                winning_trade_this_step = True
        
        # Mettre à jour l'equity
        self._update_equity()
        
        # ✅ CALCUL DE RÉCOMPENSE AVEC DSR
        if len(self.portfolio_value_history) >= 2:
            prev_equity = self.portfolio_value_history[-2]
            step_return = (self.equity - prev_equity) / prev_equity
            
            drawdown = 0.0
            if self.peak_value > 0:
                drawdown = (self.peak_value - self.equity) / self.peak_value
                
            reward = self.reward_calculator.calculate(
                step_return=step_return,
                current_drawdown=drawdown,
                trades_today=trades_executed,
                is_winning_trade=winning_trade_this_step
            )
        else:
            reward = 0.0
            
        self.current_step += 1
        
        # Condition d'arrêt
        self.done = (
            self.current_step >= self.max_steps or
            self.equity < self.initial_balance * 0.5 or
            self.balance < 0
        )
        
        obs = self._get_observation()
        
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return obs, float(reward), self.done, False, self._get_info()
    
    def _execute_trade(self, ticker: str, action: int, ticker_idx: int) -> float:
        """Exécuter une action de trading"""
        if action == 0: return 0.0
        
        if self.trades_today >= self.max_trades_per_day: return 0.0
        if (self.current_step - self.last_trade_step[ticker]) < self.min_holding_period: return 0.0
        
        current_price = self._get_current_price(ticker)
        if current_price <= 0: return 0.0
        
        if action == 1:  # BUY
            max_invest = min(self.balance * self.buy_pct, self.equity * self.max_position_pct)
            if max_invest < current_price: return 0.0
            
            execution_price = current_price * (1 + self.spread_bps)
            quantity = max_invest / execution_price
            cost = quantity * execution_price
            
            if cost <= self.balance:
                self.balance -= cost
                self.portfolio[ticker] += quantity
                self.entry_prices[ticker] = execution_price
                self.trades_today += 1
                self.total_trades += 1
                self.last_trade_step[ticker] = self.current_step
                return 0.0 # Pas de reward immédiat pour l'achat
        
        elif action == 2:  # SELL
            quantity = self.portfolio[ticker]
            if quantity < 1e-6: return 0.0
            
            execution_price = current_price * (1 - self.spread_bps)
            proceeds = quantity * execution_price
            
            pnl = 0.0
            if self.entry_prices[ticker] > 0:
                cost_basis = quantity * self.entry_prices[ticker]
                pnl = (proceeds - cost_basis) / cost_basis
                
                if pnl > 0: self.winning_trades += 1
                else: self.losing_trades += 1
            
            self.balance += proceeds
            self.portfolio[ticker] = 0.0
            self.entry_prices[ticker] = 0.0
            self.trades_today += 1
            self.total_trades += 1
            self.last_trade_step[ticker] = self.current_step
            
            # Retourne le PnL pour le bonus de win trade
            return pnl
        
        return 0.0
    
    # ... (Les méthodes helper _get_current_price, _update_equity, _get_observation restent identiques à la V6 corrigée)
    # Je les réinclus pour être sûr que le fichier soit complet et fonctionnel
    
    def _get_current_price(self, ticker: str) -> float:
        df = self.processed_data[ticker]
        if self.current_step >= len(df): return df.iloc[-1]['Close']
        price = df.iloc[self.current_step]['Close']
        if np.isnan(price) or np.isinf(price) or price <= 0: return df['Close'].median()
        return price
    
    def _update_equity(self):
        portfolio_value = 0.0
        for ticker in self.tickers:
            price = self._get_current_price(ticker)
            if price > 0: portfolio_value += self.portfolio[ticker] * price
        self.equity = self.balance + portfolio_value
        self.portfolio_value_history.append(self.equity)
        if self.equity > self.peak_value: self.peak_value = self.equity

    def _get_observation(self) -> np.ndarray:
        obs_parts = []
        for ticker in self.tickers:
            df = self.processed_data[ticker]
            if self.current_step >= len(df):
                features = np.zeros(len(self.feature_columns), dtype=np.float32)
            else:
                row = df.iloc[self.current_step]
                features = pd.to_numeric(row[self.feature_columns], errors='coerce').values.astype(np.float32)
            features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
            features = np.clip(features, -10.0, 10.0)
            obs_parts.append(features)
        
        for ticker in self.tickers:
            price = self._get_current_price(ticker)
            if price > 0:
                position_value = self.portfolio[ticker] * price
                position_pct = position_value / (self.equity + 1e-8)
            else:
                position_pct = 0.0
            position_pct = np.clip(position_pct, 0.0, 1.0)
            obs_parts.append([position_pct])
        
        cash_pct = np.clip(self.balance / (self.equity + 1e-8), 0.0, 1.0)
        total_return = np.clip((self.equity - self.initial_balance) / self.initial_balance, -1.0, 5.0)
        drawdown = np.clip((self.peak_value - self.equity) / (self.peak_value + 1e-8), 0.0, 1.0)
        
        obs_parts.append([cash_pct, total_return, drawdown])
        obs = np.concatenate([np.array(p, dtype=np.float32).flatten() for p in obs_parts])
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = np.clip(obs, -10.0, 10.0)
        return obs.astype(np.float32)
    
    def _get_info(self) -> dict:
        return {
            'equity': float(self.equity),
            'balance': float(self.balance),
            'total_return': float((self.equity - self.initial_balance) / self.initial_balance),
            'total_trades': int(self.total_trades),
            'winning_trades': int(self.winning_trades),
            'losing_trades': int(self.losing_trades),
            'current_step': int(self.current_step)
        }
