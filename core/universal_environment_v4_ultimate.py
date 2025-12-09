# core/universal_environment_v4_ultimate.py
"""Environnement V4 ULTIMATE - Performance Maximale avec Robustesse

Features:
- 37 indicateurs techniques avancés
- Normalisation robuste (IQR + Winsorization)
- Gestion complète NaN/Inf
- Slippage réaliste
- Récompenses multi-objectifs
- Architecture optimisée
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict
from collections import deque

from core.advanced_features import AdvancedFeatureEngineering


class UniversalTradingEnvV4Ultimate(gym.Env):
    """Environnement Gymnasium ULTIMATE pour trading haute performance"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        initial_balance: float = 100000.0,
        commission: float = 0.0,
        sec_fee: float = 0.0000221,
        finra_taf: float = 0.000145,
        max_steps: int = 2000,
        buy_pct: float = 0.2,
        slippage_model: str = 'realistic',
        spread_bps: float = 2.0,
        market_impact_factor: float = 0.0001,
        max_position_pct: float = 0.3,
        reward_scaling: float = 1.0,
        use_sharpe_penalty: bool = True,
        use_drawdown_penalty: bool = True,
        max_trades_per_day: int = 3,
        min_holding_period: int = 0,
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
        
        # Rewards
        self.reward_scaling = reward_scaling
        self.use_sharpe_penalty = use_sharpe_penalty
        self.use_drawdown_penalty = use_drawdown_penalty
        
        # Trading rules
        self.max_trades_per_day = max_trades_per_day
        self.min_holding_period = min_holding_period
        
        # State
        self.current_step = 0
        self.max_steps = max_steps
        self.done = False
        
        # Portfolio tracking
        self.portfolio = {ticker: 0.0 for ticker in self.tickers}
        self.portfolio_value_history = deque(maxlen=252)
        self.returns_history = deque(maxlen=100)
        self.peak_value = initial_balance
        
        # Trading tracking
        self.trades_today = 0
        self.last_trade_step = {ticker: -999 for ticker in self.tickers}
        self.total_trades = 0
        
        # ✅ FEATURES AVANCÉES avec normalisation robuste
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
        
        print(f"✅ Env ULTIMATE: {self.n_assets} tickers × {n_features_per_ticker} features = {obs_size} dims")
    
    def _prepare_advanced_features(self):
        """✅ Préparer features AVANCÉES avec normalisation ROBUSTE"""
        self.processed_data = {}
        self.feature_engineer = AdvancedFeatureEngineering()
        
        for ticker in self.tickers:
            df = self.data[ticker].copy()
            
            # Calculer features avancées (avec gestion NaN intégrée)
            df = self.feature_engineer.calculate_all_features(df)
            
            self.processed_data[ticker] = df
        
        # Feature columns (exclure OHLCV)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.feature_columns = [
            col for col in self.processed_data[self.tickers[0]].columns
            if col not in exclude_cols
        ]
        
        # Longueur minimale
        self.max_steps = min(
            self.max_steps,
            min(len(df) for df in self.processed_data.values()) - 100
        )
        
        print(f"  Features calculées: {len(self.feature_columns)}")
    
    def reset(self, seed=None, options=None):
        """Réinitialiser l'environnement"""
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_value = self.initial_balance
        
        self.portfolio = {ticker: 0.0 for ticker in self.tickers}
        self.portfolio_value_history.clear()
        self.returns_history.clear()
        
        self.current_step = np.random.randint(100, max(101, self.max_steps // 2))
        self.trades_today = 0
        self.last_trade_step = {ticker: -999 for ticker in self.tickers}
        self.total_trades = 0
        self.done = False
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, actions):
        """Exécuter une action"""
        if self.done:
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        # Reset trades counter
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
        
        # Calculer reward global
        reward = self._calculate_reward(total_reward, trades_executed)
        
        # Clip reward
        reward = np.clip(reward, -10, 10)
        
        # Next step
        self.current_step += 1
        
        # Check done
        self.done = (
            self.current_step >= self.max_steps or
            self.equity < self.initial_balance * 0.5 or
            self.balance < 0
        )
        
        terminated = self.done
        truncated = False
        
        obs = self._get_observation()
        
        # ✅ Sécurité finale
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return obs, float(reward), terminated, truncated, self._get_info()
    
    def _execute_trade(self, ticker: str, action: int, ticker_idx: int) -> float:
        """Exécuter un trade"""
        if action == 0:  # HOLD
            return 0.0
        
        if self.trades_today >= self.max_trades_per_day:
            return -0.1
        
        if (self.current_step - self.last_trade_step[ticker]) < self.min_holding_period:
            return -0.05
        
        current_price = self._get_current_price(ticker)
        
        if current_price <= 0 or np.isnan(current_price) or np.isinf(current_price):
            return -0.1
        
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
                self.trades_today += 1
                self.total_trades += 1
                self.last_trade_step[ticker] = self.current_step
                return 0.0
        
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
            
            self.balance += net_proceeds
            self.portfolio[ticker] = 0.0
            self.trades_today += 1
            self.total_trades += 1
            self.last_trade_step[ticker] = self.current_step
            return 0.0
        
        return 0.0
    
    def _apply_slippage_buy(self, ticker: str, price: float) -> float:
        """Slippage achat"""
        if self.slippage_model == 'none':
            return price
        
        slippage_pct = np.random.uniform(0.0001, 0.001)
        return price * (1 + slippage_pct)
    
    def _apply_slippage_sell(self, ticker: str, price: float) -> float:
        """Slippage vente"""
        if self.slippage_model == 'none':
            return price
        
        slippage_pct = np.random.uniform(0.0001, 0.001)
        return price * (1 - slippage_pct)
    
    def _get_current_price(self, ticker: str) -> float:
        """Prix actuel"""
        df = self.processed_data[ticker]
        if self.current_step >= len(df):
            return df.iloc[-1]['Close']
        price = df.iloc[self.current_step]['Close']
        
        if np.isnan(price) or np.isinf(price) or price <= 0:
            return df['Close'].median()
        
        return price
    
    def _update_equity(self):
        """Mettre à jour l'équité"""
        portfolio_value = 0.0
        for ticker in self.tickers:
            price = self._get_current_price(ticker)
            if price > 0:
                portfolio_value += self.portfolio[ticker] * price
        
        self.equity = self.balance + portfolio_value
        self.portfolio_value_history.append(self.equity)
        
        if self.equity > self.peak_value:
            self.peak_value = self.equity
    
    def _calculate_reward(self, trade_reward: float, trades_executed: int) -> float:
        """Reward sophistiqué"""
        if len(self.portfolio_value_history) < 2:
            return 0.0
        
        prev_equity = self.portfolio_value_history[-2]
        current_equity = self.equity
        
        if prev_equity <= 0:
            return 0.0
        
        pct_return = (current_equity - prev_equity) / prev_equity
        self.returns_history.append(pct_return)
        
        reward = pct_return * 100
        
        # Drawdown penalty
        if self.use_drawdown_penalty and self.peak_value > 0:
            drawdown = (self.peak_value - current_equity) / self.peak_value
            if drawdown > 0.1:
                reward -= drawdown * 5
        
        # Overtrading penalty
        if trades_executed > 1:
            reward -= 0.01 * trades_executed
        
        # Performance bonus
        if pct_return > 0.01:
            reward += 0.2
        
        return reward * self.reward_scaling
    
    def _get_observation(self) -> np.ndarray:
        """✅ Observation ROBUSTE"""
        obs_parts = []
        
        # Features pour chaque ticker
        for ticker in self.tickers:
            df = self.processed_data[ticker]
            
            if self.current_step >= len(df):
                features = np.zeros(len(self.feature_columns))
            else:
                row = df.iloc[self.current_step]
                features = row[self.feature_columns].values
            
            # ✅ Sécurité
            features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
            features = np.clip(features, -10, 10)
            
            obs_parts.append(features)
        
        # Portfolio state
        for ticker in self.tickers:
            price = self._get_current_price(ticker)
            if price > 0:
                position_value = self.portfolio[ticker] * price
                position_pct = position_value / (self.equity + 1e-8)
            else:
                position_pct = 0.0
            
            position_pct = np.clip(position_pct, 0, 1)
            obs_parts.append([position_pct])
        
        # Global state
        cash_pct = np.clip(self.balance / (self.equity + 1e-8), 0, 1)
        total_return = np.clip((self.equity - self.initial_balance) / self.initial_balance, -1, 5)
        drawdown = np.clip((self.peak_value - self.equity) / (self.peak_value + 1e-8), 0, 1)
        
        obs_parts.append([cash_pct, total_return, drawdown])
        
        # Concat
        obs = np.concatenate([np.array(p).flatten() for p in obs_parts])
        
        # ✅ Vérification finale
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = np.clip(obs, -10, 10)
        
        return obs.astype(np.float32)
    
    def _get_info(self) -> dict:
        """Info"""
        return {
            'equity': float(self.equity),
            'balance': float(self.balance),
            'total_return': float((self.equity - self.initial_balance) / self.initial_balance),
            'total_trades': int(self.total_trades),
            'current_step': int(self.current_step)
        }
    
    def render(self, mode='human'):
        """Afficher l'état"""
        print(f"Step: {self.current_step} | Equity: ${self.equity:,.2f} | "
              f"Return: {(self.equity/self.initial_balance - 1)*100:.2f}% | "
              f"Trades: {self.total_trades}")
