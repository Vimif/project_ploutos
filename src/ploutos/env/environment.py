#!/usr/bin/env python3
"""
Ploutos Trading Environment - Optimized Timing with Advanced Features

Objective: Resolve "buy high" and "never sell" issues to improve win rate.

Improvements:
- Optimized Feature Engineering
- Differential Sharpe Ratio (DSR) Reward
- Dynamic Support/Resistance Detection
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict
from collections import deque

from ploutos.features.pipeline import FeaturePipeline
from ploutos.env.rewards import AdvancedRewardCalculator
from sklearn.cluster import KMeans

class TradingEnvironment(gym.Env):
    """Trading Environment with Optimized Features for precise timing"""
    
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
        reward_scaling: float = 1.0,  # Reduced as DSR is self-calibrated
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
        
        self.max_trades_per_day = max_trades_per_day
        
        # Risk Management
        self.stop_loss_pct = 0.02   # -2% (1R)
        self.take_profit_pct = 0.04 # +4% (2R)
        self.entry_prices = np.zeros(0) # Placeholder, resized in reset/prep
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
        
        # Advanced Reward Calculator
        self.reward_calculator = AdvancedRewardCalculator()
        
        # Prepare Features
        self._prepare_features()
        
        # Define spaces
        n_features_per_ticker = len(self.feature_columns)
        # Obs = Features + Portfolio Weights + Cash/Equity/Time + Sector Exposures
        obs_size = self.n_assets * n_features_per_ticker + self.n_assets + 3 + self.n_sectors
        
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        self.action_space = gym.spaces.MultiDiscrete([3] * self.n_assets)
        
        print(f"Values initialized: {self.n_assets} tickers x {n_features_per_ticker} features = {obs_size} dims")
    
    def _prepare_features(self):
        """Prepare Optimized Features"""
        print(f"  [Env] Calculating Optimized Features...")
        
        self.processed_data = {}
        self.feature_engineer = FeaturePipeline()
        
        for ticker in self.tickers:
            df = self.data[ticker].copy()
            df = self.feature_engineer.calculate_all_features(df)
            self.processed_data[ticker] = df
        
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date', 'Datetime', 'Dividends', 'Stock Splits']
        
        # Get one df to check columns
        sample_df = self.processed_data[self.tickers[0]]
        
        # Select valid numeric feature columns only
        numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
        
        self.feature_columns = [
            col for col in numeric_cols
            if col not in exclude_cols
        ]
        
        print(f"  [Env] {len(self.feature_columns)} features calculated per ticker")
        
        self.max_steps = min(
            self.max_steps,
            min(len(df) for df in self.processed_data.values()) - 100
        )

        # ---------------------------------------------------------
        # OPTIMIZATION: Convert to Dense Numpy Arrays for O(1) Access
        # ---------------------------------------------------------
        print(f"  [Env] Vectorizing data for performance...")
        
        # 1. Align data lengths
        min_len = min(len(df) for df in self.processed_data.values())
        
        # 2. Pre-allocate arrays
        # Features: (Time, Ticker, Features)
        self.features_array = np.zeros((min_len, self.n_assets, len(self.feature_columns)), dtype=np.float32)
        # Prices: (Time, Ticker) - using 'Close' for simplicity
        self.prices_array = np.zeros((min_len, self.n_assets), dtype=np.float32)
        self.high_array = np.zeros((min_len, self.n_assets), dtype=np.float32)
        self.low_array = np.zeros((min_len, self.n_assets), dtype=np.float32)
        
        for i, ticker in enumerate(self.tickers):
            df = self.processed_data[ticker].iloc[:min_len]
            
            # Fill features
            feats = df[self.feature_columns].values.astype(np.float32)
            self.features_array[:, i, :] = feats
            
            # Fill prices
            self.prices_array[:, i] = df['Close'].values.astype(np.float32)
            if 'High' in df.columns and 'Low' in df.columns:
                self.high_array[:, i] = df['High'].values.astype(np.float32)
                self.low_array[:, i] = df['Low'].values.astype(np.float32)
            else:
                # Fallback if High/Low missing
                self.high_array[:, i] = self.prices_array[:, i]
                self.low_array[:, i] = self.prices_array[:, i]
            
        # Store dates for backtesting
        # Assuming all tickers share the same index after alignment (or close enough for index[0])
        self.dates = self.processed_data[self.tickers[0]].index[:min_len].values
            
        # Replace NaN/Inf in the entire array once
        self.features_array = np.nan_to_num(self.features_array, nan=0.0, posinf=10.0, neginf=-10.0)
        self.features_array = np.clip(self.features_array, -10.0, 10.0)
        
        print(f"  [Env] Data vectorized. Shape: {self.features_array.shape}, Size: {self.features_array.nbytes / 1e9:.2f} GB")
        
        # 3. CRITICAL: Clear redundant memory
        del self.processed_data
        del self.feature_engineer
        
        # 4. COMPUTE SECTOR CLUSTERS (Correlation-based)
        # ----------------------------------------------
        print(f"  [Env] Computing Correlation Clusters...")
        # Compute returns: (Times, Assets)
        returns = np.diff(self.prices_array, axis=0) / (self.prices_array[:-1] + 1e-8)
        # Handle NaN/Inf
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Transpose to (Assets, Time) for correlation
        # If too many time steps, take last 1000
        recent_returns = returns[-1000:].T if len(returns) > 1000 else returns.T
        
        # Correlation Matrix (Assets x Assets)
        # Only compute if we have enough assets
        self.n_sectors = 10
        if self.n_assets > self.n_sectors:
            corr_matrix = np.corrcoef(recent_returns)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            
            # Cluster assets based on correlation profiles
            kmeans = KMeans(n_clusters=self.n_sectors, random_state=42, n_init=10)
            self.ticker_clusters = kmeans.fit_predict(corr_matrix)
        else:
            # Fallback for few assets
            self.n_sectors = self.n_assets
            self.ticker_clusters = np.arange(self.n_assets)
            
        print(f"  [Env] Assets grouped into {self.n_sectors} correlation clusters.")
        import gc
        gc.collect()
        print(f"  [Env] Redundant data cleared from RAM.")
    
    def reset(self, seed=None, options=None):
        """Reset environment for a new episode"""
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_value = self.initial_balance
        
        # Vectorized Portfolio State
        self.portfolio_qty = np.zeros(self.n_assets, dtype=np.float32)
        self.entry_prices = np.zeros(self.n_assets, dtype=np.float32)
        self.sector_exposure = np.zeros(self.n_sectors, dtype=np.float32)
        
        self.portfolio = {ticker: 0.0 for ticker in self.tickers} # Keep for compatibility, but assume synced
        
        self.portfolio_value_history.clear()
        self.returns_history.clear()
        
        self.reward_calculator.reset()  # Reset DSR
        
        # Start Step Logic
        if options and 'current_step' in options:
            self.current_step = options['current_step']
        else:
            # Training Mode: Random Start
            # Ensure we have enough data
            if self.max_steps < 100:
                 self.current_step = 0
            else:
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
        """Execute one step in the environment"""
        if self.done:
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        # Reset trades for the day
        if self.current_step % 78 == 0:
            self.trades_today = 0
        
        trades_executed = 0
        winning_trade_this_step = False
        
        # 1. RISK MANAGEMENT CHECK (Stop Loss / Take Profit)
        # --------------------------------------------------
        skipped_tickers = self._check_risk_management()
        
        # Execute actions
        for i, (ticker, action) in enumerate(zip(self.tickers, actions)):
            if i in skipped_tickers: continue # Skip if risk exit happened
            
            trade_res = self._execute_trade(ticker, action, i)
            if action != 0:
                trades_executed += 1
            if trade_res > 0.05: # If winning trade
                winning_trade_this_step = True
        
        # Update equity
        self._update_equity()
        
        # REWARD CALCULATION WITH DSR
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
        
        # Stop condition
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
        """Execute a trading action (Vectorized)"""
        if action == 0: return 0.0
        
        if self.trades_today >= self.max_trades_per_day: return 0.0
        if (self.current_step - self.last_trade_step[ticker]) < self.min_holding_period: return 0.0
        
        # Use vectorized price array
        current_price = self.prices_array[self.current_step, ticker_idx]
        if current_price <= 0: return 0.0
        
        if action == 1:  # BUY
            max_invest = min(self.balance * self.buy_pct, self.equity * self.max_position_pct)
            if max_invest < current_price: return 0.0
            
            execution_price = current_price * (1 + self.spread_bps)
            quantity = max_invest / execution_price
            cost = quantity * execution_price
            
            if cost <= self.balance:
                self.balance -= cost
                
                # Vectorized update
                self.portfolio_qty[ticker_idx] += quantity
                self.entry_prices[ticker_idx] = execution_price
                
                # Sunc legacy dict (optional but good for debugging)
                self.portfolio[ticker] += quantity
                
                self.trades_today += 1
                self.total_trades += 1
                self.last_trade_step[ticker] = self.current_step
                return 0.0 
        
        elif action == 2:  # SELL
            # Vectorized access
            quantity = self.portfolio_qty[ticker_idx]
            if quantity < 1e-6: return 0.0
            
            execution_price = current_price * (1 - self.spread_bps)
            proceeds = quantity * execution_price
            
            pnl = 0.0
            entry_price = self.entry_prices[ticker_idx]
            if entry_price > 0:
                cost_basis = quantity * entry_price
                pnl = (proceeds - cost_basis) / cost_basis
                
                if pnl > 0: self.winning_trades += 1
                else: self.losing_trades += 1
            
            self.balance += proceeds
            
            # Vectorized update
            self.portfolio_qty[ticker_idx] = 0.0
            self.entry_prices[ticker_idx] = 0.0
            
            # Sync legacy dict
            self.portfolio[ticker] = 0.0
            
            self.trades_today += 1
            self.total_trades += 1
            self.last_trade_step[ticker] = self.current_step
            
            return pnl
        
        return 0.0
    
    def _check_risk_management(self):
        """Check and enforce Stop Loss / Take Profit"""
        triggered_indices = set()
        held_indices = np.where(self.portfolio_qty > 0)[0]
        
        if len(held_indices) == 0: return triggered_indices
        
        # Get data for held assets
        current_lows = self.low_array[self.current_step, held_indices]
        current_highs = self.high_array[self.current_step, held_indices]
        entry_prices = self.entry_prices[held_indices]
        
        # Stop Loss: Entry * (1 - SL)
        sl_levels = entry_prices * (1 - self.stop_loss_pct)
        # Take Profit: Entry * (1 + TP)
        tp_levels = entry_prices * (1 + self.take_profit_pct)
        
        for i, idx in enumerate(held_indices):
            # Check Stop Loss
            if current_lows[i] <= sl_levels[i]:
                self._force_sell(idx, sl_levels[i], "Stop Loss")
                triggered_indices.add(idx)
                
            # Check Take Profit (Elif to avoid double sell)
            elif current_highs[i] >= tp_levels[i]:
                self._force_sell(idx, tp_levels[i], "Take Profit")
                triggered_indices.add(idx)
                
        return triggered_indices

    def _force_sell(self, ticker_idx, price, reason):
        """Force exit a position at a specific price"""
        quantity = self.portfolio_qty[ticker_idx]
        if quantity <= 0: return
        
        proceeds = quantity * price
        self.balance += proceeds
        
        # Calculate PnL for stats
        entry_price = self.entry_prices[ticker_idx]
        if entry_price > 0:
            pnl = (proceeds - (quantity * entry_price)) / (quantity * entry_price)
            if pnl > 0: self.winning_trades += 1
            else: self.losing_trades += 1
            
        # Clear Position
        self.portfolio_qty[ticker_idx] = 0.0
        self.entry_prices[ticker_idx] = 0.0
        self.portfolio[self.tickers[ticker_idx]] = 0.0 # Sync
        
        self.trades_today += 1
        self.total_trades += 1
    
    # Helper methods for price retrieval, equity update, and observation generation
    
    def _get_current_price(self, ticker: str) -> float:
        # Resolve ticker index
        try:
            ticker_idx = self.tickers.index(ticker)
        except ValueError:
            return 0.0
            
        price = self.prices_array[self.current_step, ticker_idx]
        if np.isnan(price) or np.isinf(price) or price <= 0: return 1.0 # Default fallback
        return price
    
    def _update_equity(self):
        # Optimized equity calculation
        current_prices = self.prices_array[self.current_step]
        portfolio_value = np.sum(self.portfolio_qty * current_prices)
        
        self.equity = self.balance + portfolio_value
        self.portfolio_value_history.append(self.equity)
        if self.equity > self.peak_value: self.peak_value = self.equity

    def _get_observation(self) -> np.ndarray:
        # 1. Features (Vectorized Slicing) - O(1)
        # Shape: (N_Tickers, N_Features)
        current_features = self.features_array[self.current_step]
        
        # 2. Portfolio State (Vectorized) - O(1)
        # Get current prices from array
        current_prices = self.prices_array[self.current_step]
        
        # Avoid division by zero
        equity_safe = self.equity + 1e-8
        
        # Calculate position %: (Qty * Price) / Equity
        # Element-wise multiplication
        position_values = self.portfolio_qty * current_prices
        position_pcts = position_values / equity_safe
        position_pcts = np.clip(position_pcts, 0.0, 1.0)
        
        # Reshape for concatenation
        # Flatten features: (N_Tickers * N_Features,)
        features_flat = current_features.flatten()
        
        # Global Metrics
        cash_pct = np.clip(self.balance / equity_safe, 0.0, 1.0)
        total_return = np.clip((self.equity - self.initial_balance) / self.initial_balance, -1.0, 5.0)
        
        if self.peak_value > 0:
            drawdown = (self.peak_value - self.equity) / self.peak_value
        else:
            drawdown = 0.0
        drawdown = np.clip(drawdown, 0.0, 1.0)
        
        # Calculate Sector Exposure (Vectorized)
        # Sum position % by Sector ID
        sector_weights = np.bincount(self.ticker_clusters, weights=position_pcts, minlength=self.n_sectors)
        self.sector_exposure = sector_weights.astype(np.float32)

        # Concatenate all
        # [Features..., Positions..., Cash, Return, Drawdown, Sectors...]
        obs = np.concatenate([
            features_flat,
            position_pcts,
            [cash_pct, total_return, drawdown],
            self.sector_exposure
        ])
        
        # Final safety check
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
             obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
             
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
