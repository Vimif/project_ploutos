# core/universal_environment_v4_optimal.py
"""✅ Environnement V4 OPTIMAL - Configuration pour Performance Maximale

Améliorations vs V4 Ultimate:
- Rewards bonus pour trades réussis
- Pénalités réduites
- Contraintes assouplies
- Encouragement à l'action
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict
from collections import deque

from core.advanced_features import AdvancedFeatureEngineering


class UniversalTradingEnvV4Optimal(gym.Env):
    """Environnement Gymnasium OPTIMAL pour performance maximale"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        initial_balance: float = 100000.0,
        commission: float = 0.0,
        sec_fee: float = 0.0000221,
        finra_taf: float = 0.000145,
        max_steps: int = 2000,
        buy_pct: float = 0.25,
        slippage_model: str = 'realistic',
        spread_bps: float = 2.0,
        market_impact_factor: float = 0.0001,
        max_position_pct: float = 0.35,
        reward_scaling: float = 1.5,
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
        
        # Capital
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        
        # Frais
        self.commission = commission
        self.sec_fee = sec_fee
        self.finra_taf = finra_taf
        
        # Slippage
        self.slippage_model = slippage_model
        self.spread_bps = spread_bps / 10000
        self.market_impact_factor = market_impact_factor
        
        # Portfolio
        self.max_position_pct = max_position_pct
        self.buy_pct = buy_pct
        
        # ✅ Rewards OPTIMISÉS
        self.reward_scaling = reward_scaling
        self.use_sharpe_penalty = use_sharpe_penalty
        self.use_drawdown_penalty = use_drawdown_penalty
        self.reward_trade_success = reward_trade_success  # NOUVEAU
        self.penalty_overtrading = penalty_overtrading     # Réduit
        self.drawdown_penalty_factor = drawdown_penalty_factor  # Réduit
        
        # Trading rules (assouplis)
        self.max_trades_per_day = max_trades_per_day
        self.min_holding_period = min_holding_period
        
        # State
        self.current_step = 0
        self.max_steps = max_steps
        self.done = False
        
        # Portfolio tracking
        self.portfolio = {ticker: 0.0 for ticker in self.tickers}
        self.entry_prices = {ticker: 0.0 for ticker in self.tickers}
        self.portfolio_value_history = deque(maxlen=252)
        self.returns_history = deque(maxlen=100)
        self.peak_value = initial_balance
        
        # Trading tracking
        self.trades_today = 0
        self.last_trade_step = {ticker: -999 for ticker in self.tickers}
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Features avancées
        self._prepare_advanced_features()
        
        # Spaces
        n_features_per_ticker = len(self.feature_columns)
        obs_size = self.n_assets * n_features_per_ticker + self.n_assets + 3
        
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        self.action_space = gym.spaces.MultiDiscrete([3] * self.n_assets)
        
        print(f"✅ Env OPTIMAL: {self.n_assets} tickers × {n_features_per_ticker} features = {obs_size} dims")
    
    def _prepare_advanced_features(self):
        """Préparer features avancées"""
        self.processed_data = {}
        self.feature_engineer = AdvancedFeatureEngineering()
        
        for ticker in self.tickers:
            df = self.data[ticker].copy()
            df = self.feature_engineer.calculate_all_features(df)
            self.processed_data[ticker] = df
        
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.feature_columns = [
            col for col in self.processed_data[self.tickers[0]].columns
            if col not in exclude_cols
        ]
        
        self.max_steps = min(
            self.max_steps,
            min(len(df) for df in self.processed_data.values()) - 100
        )
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_value = self.initial_balance
        
        self.portfolio = {ticker: 0.0 for ticker in self.tickers}
        self.entry_prices = {ticker: 0.0 for ticker in self.tickers}
        self.portfolio_value_history.clear()
        self.returns_history.clear()
        
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
        if self.done:
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        if self.current_step % 78 == 0:
            self.trades_today = 0
        
        # Exécuter trades
        total_reward = 0.0
        trades_executed = 0
        
        for i, (ticker, action) in enumerate(zip(self.tickers, actions)):
            reward = self._execute_trade(ticker, action, i)
            total_reward += reward
            if action != 0:
                trades_executed += 1
        
        # Update equity
        self._update_equity()
        
        # ✅ Reward OPTIMISÉ
        reward = self._calculate_reward_optimal(total_reward, trades_executed)
        
        reward = np.clip(reward, -10, 10)
        
        self.current_step += 1
        
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
        if action == 0:  # HOLD
            return 0.0
        
        if self.trades_today >= self.max_trades_per_day:
            return -0.05  # ✅ Pénalité réduite
        
        if (self.current_step - self.last_trade_step[ticker]) < self.min_holding_period:
            return -0.02  # ✅ Pénalité réduite
        
        current_price = self._get_current_price(ticker)
        
        if current_price <= 0 or np.isnan(current_price) or np.isinf(current_price):
            return -0.05
        
        if action == 1:  # BUY
            max_invest = min(
                self.balance * self.buy_pct,
                self.equity * self.max_position_pct
            )
            
            if max_invest < current_price * 1.1:
                return -0.01
            
            execution_price = self._apply_slippage_buy(ticker, current_price)
            execution_price *= (1 + self.spread_bps)
            
            quantity = max_invest / execution_price
            cost = quantity * execution_price
            
            sec_fee_cost = cost * self.sec_fee
            finra_fee_cost = cost * self.finra_taf
            total_cost = cost + sec_fee_cost + finra_fee_cost
            
            if total_cost <= self.balance:
                self.balance -= total_cost
                self.portfolio[ticker] += quantity
                self.entry_prices[ticker] = execution_price  # ✅ Mémoriser prix d'entrée
                self.trades_today += 1
                self.total_trades += 1
                self.last_trade_step[ticker] = self.current_step
                return 0.1  # ✅ Petit bonus pour action
        
        elif action == 2:  # SELL
            quantity = self.portfolio[ticker]
            
            if quantity < 1e-6:
                return -0.01
            
            execution_price = self._apply_slippage_sell(ticker, current_price)
            execution_price *= (1 - self.spread_bps)
            
            proceeds = quantity * execution_price
            
            sec_fee_cost = proceeds * self.sec_fee
            finra_fee_cost = proceeds * self.finra_taf
            net_proceeds = proceeds - sec_fee_cost - finra_fee_cost
            
            # ✅ Calculer PnL
            pnl = 0.0
            if self.entry_prices[ticker] > 0:
                cost_basis = quantity * self.entry_prices[ticker]
                pnl = (net_proceeds - cost_basis) / cost_basis
                
                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
            
            self.balance += net_proceeds
            self.portfolio[ticker] = 0.0
            self.entry_prices[ticker] = 0.0
            self.trades_today += 1
            self.total_trades += 1
            self.last_trade_step[ticker] = self.current_step
            
            # ✅ BONUS pour trade réussi
            if pnl > 0.01:  # +1%
                return self.reward_trade_success
            
            return 0.0
        
        return 0.0
    
    def _calculate_reward_optimal(self, trade_reward: float, trades_executed: int) -> float:
        """✅ Reward OPTIMISÉ pour encourager trading"""
        if len(self.portfolio_value_history) < 2:
            return 0.0
        
        prev_equity = self.portfolio_value_history[-2]
        current_equity = self.equity
        
        if prev_equity <= 0:
            return 0.0
        
        pct_return = (current_equity - prev_equity) / prev_equity
        self.returns_history.append(pct_return)
        
        reward = pct_return * 100
        
        # ✅ Drawdown penalty RÉDUITE
        if self.use_drawdown_penalty and self.peak_value > 0:
            drawdown = (self.peak_value - current_equity) / self.peak_value
            if drawdown > 0.15:  # Seuil augmenté (0.10 → 0.15)
                reward -= drawdown * self.drawdown_penalty_factor
        
        # ✅ Overtrading penalty RÉDUITE
        if trades_executed > 2:
            reward -= self.penalty_overtrading * trades_executed
        
        # ✅ BONUS pour bonne perf
        if pct_return > 0.01:
            reward += 0.3  # Augmenté (0.2 → 0.3)
        
        # ✅ NOUVEAU: Bonus pour win rate
        if self.total_trades > 10:
            win_rate = self.winning_trades / self.total_trades
            if win_rate > 0.6:
                reward += 0.2
        
        return reward * self.reward_scaling
    
    def _apply_slippage_buy(self, ticker: str, price: float) -> float:
        if self.slippage_model == 'none':
            return price
        slippage_pct = np.random.uniform(0.0001, 0.001)
        return price * (1 + slippage_pct)
    
    def _apply_slippage_sell(self, ticker: str, price: float) -> float:
        if self.slippage_model == 'none':
            return price
        slippage_pct = np.random.uniform(0.0001, 0.001)
        return price * (1 - slippage_pct)
    
    def _get_current_price(self, ticker: str) -> float:
        df = self.processed_data[ticker]
        if self.current_step >= len(df):
            return df.iloc[-1]['Close']
        price = df.iloc[self.current_step]['Close']
        
        if np.isnan(price) or np.isinf(price) or price <= 0:
            return df['Close'].median()
        
        return price
    
    def _update_equity(self):
        portfolio_value = 0.0
        for ticker in self.tickers:
            price = self._get_current_price(ticker)
            if price > 0:
                portfolio_value += self.portfolio[ticker] * price
        
        self.equity = self.balance + portfolio_value
        self.portfolio_value_history.append(self.equity)
        
        if self.equity > self.peak_value:
            self.peak_value = self.equity
    
    def _get_observation(self) -> np.ndarray:
        obs_parts = []
        
        for ticker in self.tickers:
            df = self.processed_data[ticker]
            
            if self.current_step >= len(df):
                features = np.zeros(len(self.feature_columns))
            else:
                row = df.iloc[self.current_step]
                features = row[self.feature_columns].values
            
            features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
            features = np.clip(features, -10, 10)
            
            obs_parts.append(features)
        
        for ticker in self.tickers:
            price = self._get_current_price(ticker)
            if price > 0:
                position_value = self.portfolio[ticker] * price
                position_pct = position_value / (self.equity + 1e-8)
            else:
                position_pct = 0.0
            
            position_pct = np.clip(position_pct, 0, 1)
            obs_parts.append([position_pct])
        
        cash_pct = np.clip(self.balance / (self.equity + 1e-8), 0, 1)
        total_return = np.clip((self.equity - self.initial_balance) / self.initial_balance, -1, 5)
        drawdown = np.clip((self.peak_value - self.equity) / (self.peak_value + 1e-8), 0, 1)
        
        obs_parts.append([cash_pct, total_return, drawdown])
        
        obs = np.concatenate([np.array(p).flatten() for p in obs_parts])
        
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = np.clip(obs, -10, 10)
        
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
    
    def render(self, mode='human'):
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        print(f"Step: {self.current_step} | Equity: ${self.equity:,.2f} | "
              f"Return: {(self.equity/self.initial_balance - 1)*100:.2f}% | "
              f"Trades: {self.total_trades} (WR: {win_rate:.1%})")
