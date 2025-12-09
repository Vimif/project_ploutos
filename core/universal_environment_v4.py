# core/universal_environment_v4.py
"""Environnement de Trading V4 ULTRA-RÉALISTE - Ploutos Ultimate

Améliorations:
- Slippage réaliste basé sur la volatilité
- Spread bid/ask dynamique
- Market impact modeling
- Frais de transaction Alpaca réels
- Pattern Day Trading rules
- Récompenses basées sur Sharpe ratio
- Penalties pour drawdown
- Multi-asset portfolio
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import deque

from core.advanced_features import AdvancedFeatureEngineering, add_market_regime_features


class UniversalTradingEnvV4(gym.Env):
    """Environnement Gymnasium pour trading multi-assets ultra-réaliste"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        initial_balance: float = 100000.0,
        commission: float = 0.0,  # Alpaca commission-free
        sec_fee: float = 0.0000221,  # SEC fee
        finra_taf: float = 0.000145,  # FINRA TAF
        max_steps: int = 2000,
        buy_pct: float = 0.2,
        slippage_model: str = 'realistic',  # 'none', 'simple', 'realistic'
        spread_bps: float = 2.0,  # Spread en basis points
        market_impact_factor: float = 0.0001,
        max_position_pct: float = 0.3,  # Max 30% par action
        reward_scaling: float = 1.0,
        use_sharpe_penalty: bool = True,
        use_drawdown_penalty: bool = True,
        max_trades_per_day: int = 3,  # PDT rule
        min_holding_period: int = 0,  # Holding minimum
    ):
        super().__init__()
        
        self.data = data
        self.tickers = list(data.keys())
        self.n_assets = len(self.tickers)
        
        # Capital
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity = initial_balance
        
        # Frais réalistes Alpaca
        self.commission = commission  # 0 pour Alpaca
        self.sec_fee = sec_fee  # 0.00221% SEC
        self.finra_taf = finra_taf  # 0.0145% FINRA TAF
        
        # Slippage & Spread
        self.slippage_model = slippage_model
        self.spread_bps = spread_bps / 10000  # Convertir en decimal
        self.market_impact_factor = market_impact_factor
        
        # Portfolio constraints
        self.max_position_pct = max_position_pct
        self.buy_pct = buy_pct
        
        # Reward engineering
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
        
        # Portfolio
        self.portfolio = {ticker: 0.0 for ticker in self.tickers}
        self.portfolio_value_history = deque(maxlen=252)  # 1 an
        self.returns_history = deque(maxlen=100)
        self.peak_value = initial_balance
        
        # Tracking
        self.trades_today = 0
        self.last_trade_step = {ticker: -999 for ticker in self.tickers}
        self.total_trades = 0
        
        # Features engineering
        self.feature_engineer = AdvancedFeatureEngineering()
        self._prepare_data()
        
        # Spaces
        n_features = len(self.feature_columns)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_assets * n_features + self.n_assets + 3,),
            dtype=np.float32
        )
        
        # Actions: 0=HOLD, 1=BUY, 2=SELL pour chaque asset
        self.action_space = gym.spaces.MultiDiscrete([3] * self.n_assets)
    
    def _prepare_data(self):
        """Préparer les données avec features avancées"""
        self.processed_data = {}
        
        for ticker in self.tickers:
            df = self.data[ticker].copy()
            
            # Features avancées
            df = self.feature_engineer.calculate_all_features(df)
            
            # Features de régime
            df = add_market_regime_features(df)
            
            # Normaliser
            df = self._normalize_features(df)
            
            self.processed_data[ticker] = df
        
        # Feature columns (exclure OHLCV)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        self.feature_columns = [
            col for col in self.processed_data[self.tickers[0]].columns
            if col not in exclude_cols
        ]
        
        # Longueur minimale
        self.max_steps = min(
            self.max_steps,
            min(len(df) for df in self.processed_data.values()) - 100
        )
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliser les features"""
        # Features à ne pas normaliser
        no_norm = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        
        for col in df.columns:
            if col not in no_norm:
                # Normalisation robuste
                median = df[col].rolling(window=252, min_periods=50).median()
                mad = (df[col] - median).abs().rolling(window=252, min_periods=50).median()
                df[col] = (df[col] - median) / (mad + 1e-8)
                df[col] = df[col].clip(-10, 10)  # Clip outliers
        
        return df
    
    def reset(self, seed=None, options=None):
        """Réinitialiser l'environnement"""
        super().reset(seed=seed)
        
        # Reset capital
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_value = self.initial_balance
        
        # Reset portfolio
        self.portfolio = {ticker: 0.0 for ticker in self.tickers}
        self.portfolio_value_history.clear()
        self.returns_history.clear()
        
        # Reset tracking
        self.current_step = np.random.randint(100, self.max_steps // 2)
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
        
        # Reset trades counter (nouveau jour si step % 390 == 0 pour trading horaire)
        if self.current_step % 78 == 0:  # ~1 jour de trading
            self.trades_today = 0
        
        # Exécuter trades
        total_reward = 0.0
        trades_executed = 0
        
        for i, (ticker, action) in enumerate(zip(self.tickers, actions)):
            reward = self._execute_trade(ticker, action, i)
            total_reward += reward
            if action != 0:  # Si trade exécuté
                trades_executed += 1
        
        # Update equity
        self._update_equity()
        
        # Calculer reward global
        reward = self._calculate_reward(total_reward, trades_executed)
        
        # Next step
        self.current_step += 1
        
        # Check done
        self.done = (
            self.current_step >= self.max_steps or
            self.equity < self.initial_balance * 0.5 or  # Stop-loss 50%
            self.balance < 0
        )
        
        terminated = self.done
        truncated = False
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _execute_trade(self, ticker: str, action: int, ticker_idx: int) -> float:
        """Exécuter un trade avec slippage et fees réalistes"""
        if action == 0:  # HOLD
            return 0.0
        
        # Check PDT rule
        if self.trades_today >= self.max_trades_per_day:
            return -0.1  # Penalty
        
        # Check holding period
        if (self.current_step - self.last_trade_step[ticker]) < self.min_holding_period:
            return -0.05  # Penalty
        
        # Prix de base
        current_price = self._get_current_price(ticker)
        
        if action == 1:  # BUY
            # Calculer quantité
            max_invest = min(
                self.balance * self.buy_pct,
                self.equity * self.max_position_pct
            )
            
            if max_invest < current_price * 1.1:  # Minimum 1 action + fees
                return -0.01  # Penalty pour tentative invalid
            
            # Prix avec slippage + spread
            execution_price = self._apply_slippage_buy(ticker, current_price)
            execution_price *= (1 + self.spread_bps)  # Ask price
            
            # Quantité
            quantity = max_invest / execution_price
            cost = quantity * execution_price
            
            # Fees
            sec_fee_cost = cost * self.sec_fee
            finra_fee_cost = cost * self.finra_taf
            total_cost = cost + sec_fee_cost + finra_fee_cost
            
            if total_cost <= self.balance:
                self.balance -= total_cost
                self.portfolio[ticker] += quantity
                self.trades_today += 1
                self.total_trades += 1
                self.last_trade_step[ticker] = self.current_step
                
                # Reward proportionnel au montant
                return 0.0  # Neutre, reward viendra du P&L
        
        elif action == 2:  # SELL
            quantity = self.portfolio[ticker]
            
            if quantity < 1e-6:
                return -0.01  # Penalty pour tentative invalid
            
            # Prix avec slippage + spread
            execution_price = self._apply_slippage_sell(ticker, current_price)
            execution_price *= (1 - self.spread_bps)  # Bid price
            
            # Vente
            proceeds = quantity * execution_price
            
            # Fees
            sec_fee_cost = proceeds * self.sec_fee
            finra_fee_cost = proceeds * self.finra_taf
            net_proceeds = proceeds - sec_fee_cost - finra_fee_cost
            
            self.balance += net_proceeds
            self.portfolio[ticker] = 0.0
            self.trades_today += 1
            self.total_trades += 1
            self.last_trade_step[ticker] = self.current_step
            
            # Reward = P&L
            # (sera capturé dans le calcul global)
            return 0.0
        
        return 0.0
    
    def _apply_slippage_buy(self, ticker: str, price: float) -> float:
        """Appliquer slippage réaliste sur achat"""
        if self.slippage_model == 'none':
            return price
        
        if self.slippage_model == 'simple':
            return price * (1 + np.random.uniform(0, 0.001))
        
        # Realistic slippage basé sur ATR
        df = self.processed_data[ticker]
        if self.current_step >= len(df):
            return price
        
        volatility = df.iloc[self.current_step].get('ATR_14', 0.01)
        normalized_vol = volatility / (price + 1e-8)
        
        # Slippage proportionnel à la volatilité
        slippage_pct = normalized_vol * np.random.uniform(0.1, 0.3)
        
        return price * (1 + slippage_pct)
    
    def _apply_slippage_sell(self, ticker: str, price: float) -> float:
        """Appliquer slippage réaliste sur vente"""
        if self.slippage_model == 'none':
            return price
        
        if self.slippage_model == 'simple':
            return price * (1 - np.random.uniform(0, 0.001))
        
        # Realistic slippage
        df = self.processed_data[ticker]
        if self.current_step >= len(df):
            return price
        
        volatility = df.iloc[self.current_step].get('ATR_14', 0.01)
        normalized_vol = volatility / (price + 1e-8)
        
        slippage_pct = normalized_vol * np.random.uniform(0.1, 0.3)
        
        return price * (1 - slippage_pct)
    
    def _get_current_price(self, ticker: str) -> float:
        """Obtenir le prix actuel"""
        df = self.processed_data[ticker]
        if self.current_step >= len(df):
            return df.iloc[-1]['Close']
        return df.iloc[self.current_step]['Close']
    
    def _update_equity(self):
        """Mettre à jour l'équité totale"""
        portfolio_value = sum(
            self.portfolio[ticker] * self._get_current_price(ticker)
            for ticker in self.tickers
        )
        
        self.equity = self.balance + portfolio_value
        self.portfolio_value_history.append(self.equity)
        
        # Update peak
        if self.equity > self.peak_value:
            self.peak_value = self.equity
    
    def _calculate_reward(self, trade_reward: float, trades_executed: int) -> float:
        """Calculer reward sophistiqué"""
        if len(self.portfolio_value_history) < 2:
            return 0.0
        
        # 1. Rendement
        prev_equity = self.portfolio_value_history[-2]
        current_equity = self.equity
        pct_return = (current_equity - prev_equity) / (prev_equity + 1e-8)
        
        self.returns_history.append(pct_return)
        
        reward = pct_return * 100  # Amplifier
        
        # 2. Penalty pour drawdown
        if self.use_drawdown_penalty:
            drawdown = (self.peak_value - current_equity) / (self.peak_value + 1e-8)
            if drawdown > 0.1:  # Si drawdown > 10%
                reward -= drawdown * 10
        
        # 3. Sharpe penalty
        if self.use_sharpe_penalty and len(self.returns_history) >= 30:
            returns_array = np.array(self.returns_history)
            sharpe = returns_array.mean() / (returns_array.std() + 1e-8)
            
            if sharpe < 0:  # Pénaliser Sharpe négatif
                reward += sharpe * 0.1
        
        # 4. Penalty pour overtrading
        if trades_executed > 1:
            reward -= 0.01 * trades_executed
        
        # 5. Bonus pour bonne perf
        if pct_return > 0.02:  # +2%
            reward += 0.5
        
        return reward * self.reward_scaling
    
    def _get_observation(self) -> np.ndarray:
        """Construire l'observation"""
        obs_parts = []
        
        # Features pour chaque ticker
        for ticker in self.tickers:
            df = self.processed_data[ticker]
            
            if self.current_step >= len(df):
                features = np.zeros(len(self.feature_columns))
            else:
                features = df.iloc[self.current_step][self.feature_columns].values
            
            obs_parts.append(features)
        
        # Portfolio state
        for ticker in self.tickers:
            position_value = self.portfolio[ticker] * self._get_current_price(ticker)
            position_pct = position_value / (self.equity + 1e-8)
            obs_parts.append([position_pct])
        
        # Global state
        cash_pct = self.balance / (self.equity + 1e-8)
        total_return = (self.equity - self.initial_balance) / self.initial_balance
        
        # Drawdown
        drawdown = (self.peak_value - self.equity) / (self.peak_value + 1e-8)
        
        obs_parts.append([cash_pct, total_return, drawdown])
        
        # Concat
        obs = np.concatenate([np.array(p).flatten() for p in obs_parts])
        
        return obs.astype(np.float32)
    
    def _get_info(self) -> dict:
        """Info supplémentaires"""
        return {
            'equity': self.equity,
            'balance': self.balance,
            'total_return': (self.equity - self.initial_balance) / self.initial_balance,
            'total_trades': self.total_trades,
            'current_step': self.current_step
        }
    
    def render(self, mode='human'):
        """Afficher l'état"""
        print(f"Step: {self.current_step} | Equity: ${self.equity:,.2f} | "
              f"Return: {(self.equity/self.initial_balance - 1)*100:.2f}% | "
              f"Trades: {self.total_trades}")
