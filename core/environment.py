# core/environment.py
"""Environnement V9 - High Performance & Shared Memory

Améliorations V9:
- Support Shared Memory (Zero-Copy) pour entraînement multi-process ultra-rapide.
- Intégration FeatureEngineer V9 (Polars Optimized).
- Compatible RecurrentPPO (LSTM) et PPO standard.
- Données macroéconomiques intégrées.

Architecture:
- Données centralisées en RAM ou SharedMemory.
- State management optimisé.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Optional
from collections import deque

from core.constants import (
    BANKRUPTCY_THRESHOLD,
    EQUITY_EPSILON,
    MAX_REWARD_CLIP,
    MIN_POSITION_THRESHOLD,
    PORTFOLIO_HISTORY_WINDOW,
    RETURNS_HISTORY_WINDOW,
)
from core.env_config import EnvConfig
from core.features import FeatureEngineer
from core.macro_data import MacroDataFetcher
from core.observation_builder import ObservationBuilder
from core.reward_calculator import RewardCalculator
from core.transaction_costs import AdvancedTransactionModel

try:
    from core.shared_memory_manager import load_shared_data
    SHARED_MEMORY_AVAILABLE = True
except ImportError:
    SHARED_MEMORY_AVAILABLE = False

VALID_MODES = ("train", "eval", "backtest")


class TradingEnv(gym.Env):
    """Environnement V9 avec Shared Memory et Support LSTM.

    Modes:
        train:    Random start, slippage stochastique. Pour PPO/RecurrentPPO.
        eval:     Start fixe (step 100), slippage moyen. Pour EvalCallback.
        backtest: Start 0, AdvancedTransactionModel, seed fixé.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        macro_data: Optional[pd.DataFrame] = None,
        initial_balance: float = 100000.0,
        commission: float = 0.0,
        sec_fee: float = 0.0000221,
        finra_taf: float = 0.000145,
        max_steps: int = 2500,
        buy_pct: float = 0.20,
        slippage_model: str = "realistic",
        spread_bps: float = 2.0,
        market_impact_factor: float = 0.0001,
        max_position_pct: float = 0.25,
        reward_scaling: float = 1.5,
        use_sharpe_penalty: bool = True,
        use_drawdown_penalty: bool = True,
        max_trades_per_day: int = 10,
        min_holding_period: int = 2,
        reward_trade_success: float = 0.5,
        penalty_overtrading: float = 0.005,
        drawdown_penalty_factor: float = 3.0,
        mode: str = "train",
        seed: Optional[int] = None,
        reward_buy_executed: float = 0.1,
        reward_overtrading: float = -0.02,
        reward_invalid_trade: float = -0.01,
        reward_bad_price: float = -0.05,
        reward_good_return_bonus: float = 0.3,
        reward_high_winrate_bonus: float = 0.2,
        good_return_threshold: float = 0.01,
        high_winrate_threshold: float = 0.6,
        features_precomputed: bool = False,  # Nouveau flag
        max_features_per_ticker: int = 0,  # 0 = all features, >0 = top N by variance
        warmup_steps: int = 100,
        steps_per_trading_week: int = 78,
        drawdown_threshold: float = 0.10,
    ):
        super().__init__()

        if mode not in VALID_MODES:
            raise ValueError(f"mode doit être l'un de {VALID_MODES}, obtenu '{mode}'")
        self.mode = mode
        self.features_precomputed = features_precomputed  # Stocker le flag
        self.max_features_per_ticker = max_features_per_ticker

        self._seed = seed
        self._rng = np.random.RandomState(seed)

        # V9 Shared Memory Auto-Detect
        self._shm_objects = []
        if SHARED_MEMORY_AVAILABLE and isinstance(data, dict) and len(data) > 0:
            first_val = next(iter(data.values()))
            if isinstance(first_val, dict) and "shm_name" in first_val:
                print("⚡ V9: Loading environment data from Shared Memory...")
                self.data = load_shared_data(data)
                self.features_precomputed = True # Force precomputed (SHM is read-only usually)
            else:
                self.data = data
        else:
            self.data = data

        self.tickers = list(self.data.keys())
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
        self.use_sharpe_penalty = use_sharpe_penalty
        self.use_drawdown_penalty = use_drawdown_penalty
        self.reward_trade_success = reward_trade_success
        self.penalty_overtrading = penalty_overtrading
        self.drawdown_penalty_factor = drawdown_penalty_factor

        self.reward_buy_executed = reward_buy_executed
        self.reward_overtrading_immediate = reward_overtrading
        self.reward_invalid_trade = reward_invalid_trade
        self.reward_bad_price = reward_bad_price
        self.reward_good_return_bonus = reward_good_return_bonus
        self.reward_high_winrate_bonus = reward_high_winrate_bonus
        self.good_return_threshold = good_return_threshold
        self.high_winrate_threshold = high_winrate_threshold

        self.max_trades_per_day = max_trades_per_day
        self.min_holding_period = min_holding_period
        self.warmup_steps = warmup_steps
        self.steps_per_trading_week = steps_per_trading_week
        self.drawdown_threshold = drawdown_threshold

        self.current_step = 0
        self.max_steps = max_steps
        self.done = False

        self.portfolio = {ticker: 0.0 for ticker in self.tickers}
        self.entry_prices = {ticker: 0.0 for ticker in self.tickers}
        self.portfolio_value_history = deque(maxlen=PORTFOLIO_HISTORY_WINDOW)
        self.returns_history = deque(maxlen=RETURNS_HISTORY_WINDOW)
        self.peak_value = initial_balance

        self.trades_today = 0
        self.last_trade_step = {ticker: -999 for ticker in self.tickers}
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

        self.transaction_model = AdvancedTransactionModel(
            base_commission=commission,
            market_impact_coef=market_impact_factor,
            rng=self._rng,
        )

        # Reward calculator (DSR + penalties)
        self.reward_calculator = RewardCalculator(
            reward_scaling=reward_scaling,
            use_drawdown_penalty=use_drawdown_penalty,
            drawdown_penalty_factor=drawdown_penalty_factor,
            drawdown_threshold=drawdown_threshold,
            penalty_overtrading=penalty_overtrading,
        )

        # Préparer features techniques + macro
        self.macro_data = macro_data
        self._prepare_features(macro_data)

        # Observation builder
        self.n_macro_features = len(self.macro_columns) if self.macro_columns else 0
        self.obs_builder = ObservationBuilder(
            tickers=self.tickers,
            feature_columns=self.feature_columns,
            feature_arrays=self.feature_arrays,
            macro_array=self.macro_array,
            n_macro_features=self.n_macro_features,
        )

        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(self.obs_builder.obs_size,), dtype=np.float32
        )
        self.action_space = gym.spaces.MultiDiscrete([3] * self.n_assets)

        n_features_per_ticker = len(self.feature_columns)
        mode_label = {"train": "Training", "eval": "Evaluation", "backtest": "Backtest"}
        print(
            f"Env V9 [Memory Optimized] [{mode_label[self.mode]}]: "
            f"{self.n_assets} tickers x {n_features_per_ticker} features "
            f"+ {self.n_macro_features} macro = {self.obs_builder.obs_size} dims"
        )

    @classmethod
    def from_config(
        cls,
        config: EnvConfig,
        data: Dict[str, pd.DataFrame],
        macro_data: Optional[pd.DataFrame] = None,
        mode: str = "train",
        seed: Optional[int] = None,
    ) -> "TradingEnv":
        """Create a TradingEnv from a structured EnvConfig dataclass."""
        tx = config.transaction
        rw = config.reward
        tr = config.trading
        return cls(
            data=data,
            macro_data=macro_data,
            initial_balance=config.initial_balance,
            commission=tx.commission,
            sec_fee=tx.sec_fee,
            finra_taf=tx.finra_taf,
            slippage_model=tx.slippage_model,
            spread_bps=tx.spread_bps,
            market_impact_factor=tx.market_impact_factor,
            max_steps=tr.max_steps,
            buy_pct=tr.buy_pct,
            max_position_pct=tr.max_position_pct,
            max_trades_per_day=tr.max_trades_per_day,
            min_holding_period=tr.min_holding_period,
            warmup_steps=tr.warmup_steps,
            steps_per_trading_week=tr.steps_per_trading_week,
            drawdown_threshold=tr.drawdown_threshold,
            reward_scaling=rw.reward_scaling,
            use_sharpe_penalty=rw.use_sharpe_penalty,
            use_drawdown_penalty=rw.use_drawdown_penalty,
            reward_trade_success=rw.reward_trade_success,
            penalty_overtrading=rw.penalty_overtrading,
            drawdown_penalty_factor=rw.drawdown_penalty_factor,
            reward_buy_executed=rw.reward_buy_executed,
            reward_overtrading=rw.reward_overtrading,
            reward_invalid_trade=rw.reward_invalid_trade,
            reward_bad_price=rw.reward_bad_price,
            reward_good_return_bonus=rw.reward_good_return_bonus,
            reward_high_winrate_bonus=rw.reward_high_winrate_bonus,
            good_return_threshold=rw.good_return_threshold,
            high_winrate_threshold=rw.high_winrate_threshold,
            features_precomputed=config.features_precomputed,
            max_features_per_ticker=config.max_features_per_ticker,
            mode=mode,
            seed=seed,
        )

    def _prepare_features(self, macro_data: Optional[pd.DataFrame]):
        """Préparer Features V2 + macro."""
        self.processed_data = {}
        self.feature_engineer = FeatureEngineer()
        self.macro_fetcher = MacroDataFetcher()

        for ticker in self.tickers:
            df = self.data[ticker].copy()  # Copie locale légère

            if not self.features_precomputed:
                # Calcul COÛTEUX (seulement si non pré-calculé)
                df = self.feature_engineer.calculate_all_features(df)

            self.processed_data[ticker] = df

        exclude_cols = {"Open", "High", "Low", "Close", "Volume", "Date", "Datetime", "Timestamp"}
        ref_df = self.processed_data[self.tickers[0]]
        all_feature_cols = [
            col
            for col in ref_df.columns
            if col not in exclude_cols and ref_df[col].dtype in (np.float64, np.float32, np.int64)
        ]

        # Feature selection by variance (reduces dimensionality)
        if self.max_features_per_ticker > 0 and len(all_feature_cols) > self.max_features_per_ticker:
            ref_df = self.processed_data[self.tickers[0]]
            variances = ref_df[all_feature_cols].var().fillna(0)
            top_features = variances.nlargest(self.max_features_per_ticker).index.tolist()
            self.feature_columns = top_features
        else:
            self.feature_columns = all_feature_cols

        # Macro : aligner sur le premier ticker comme référence
        self.macro_columns = []
        self.macro_array = None

        if macro_data is not None and not macro_data.empty:
            ref_df = self.processed_data[self.tickers[0]]
            aligned = self.macro_fetcher.align_to_ticker(macro_data, ref_df)

            if not aligned.empty:
                self.macro_columns = list(aligned.columns)
                self.macro_array = aligned.values.astype(np.float32)
                print(
                    f"  Macro features: {len(self.macro_columns)} ({', '.join(self.macro_columns[:5])}...)"
                )

        print(f"  {len(self.feature_columns)} features/ticker + {len(self.macro_columns)} macro")

        # Convertir en numpy
        self.feature_arrays = {}
        self.close_prices = {}
        self.volume_arrays = {}

        for ticker in self.tickers:
            df = self.processed_data[ticker]
            self.feature_arrays[ticker] = df[self.feature_columns].values.astype(np.float32)
            self.close_prices[ticker] = df["Close"].values.astype(np.float32)
            if "Volume" in df.columns:
                self.volume_arrays[ticker] = df["Volume"].values.astype(np.float64)
            else:
                self.volume_arrays[ticker] = np.full(len(df), 1_000_000.0)

        self.max_steps = min(
            self.max_steps,
            min(len(df) for df in self.processed_data.values()) - self.warmup_steps,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.RandomState(seed)
            self.transaction_model._rng = self._rng
        elif self.mode == "backtest" and self._seed is not None:
            self._rng = np.random.RandomState(self._seed)
            self.transaction_model._rng = self._rng

        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_value = self.initial_balance

        self.portfolio = {ticker: 0.0 for ticker in self.tickers}
        self.entry_prices = {ticker: 0.0 for ticker in self.tickers}
        self.portfolio_value_history.clear()
        self.returns_history.clear()

        self.reward_calculator.reset()

        if self.mode == "train":
            self.current_step = self._rng.randint(
                self.warmup_steps, max(self.warmup_steps + 1, self.max_steps // 2)
            )
        elif self.mode == "eval":
            self.current_step = self.warmup_steps
        elif self.mode == "backtest":
            self.current_step = 0

        self.trades_today = 0
        self.last_trade_step = {ticker: -999 for ticker in self.tickers}
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.done = False

        obs = self._get_observation()
        return obs, self._get_info()

    def step(self, actions):
        if self.done:
            return self._get_observation(), 0.0, True, False, self._get_info()

        if self.current_step % self.steps_per_trading_week == 0:
            self.trades_today = 0

        total_reward = 0.0
        trades_executed = 0

        for i, (ticker, action) in enumerate(zip(self.tickers, actions)):
            reward = self._execute_trade(ticker, action, i)
            total_reward += reward
            if action != 0:
                trades_executed += 1

        self._update_equity()
        reward = self._calculate_reward(total_reward, trades_executed)
        reward = np.clip(reward, -MAX_REWARD_CLIP, MAX_REWARD_CLIP)

        self.current_step += 1

        self.done = (
            self.current_step >= self.max_steps
            or self.equity < self.initial_balance * BANKRUPTCY_THRESHOLD
            or self.balance < 0
        )

        obs = self._get_observation()
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

        return obs, float(reward), self.done, False, self._get_info()

    def _execute_trade(self, ticker: str, action: int, ticker_idx: int) -> float:
        if action == 0:
            return 0.0

        if self.trades_today >= self.max_trades_per_day:
            return self.reward_overtrading_immediate

        if (self.current_step - self.last_trade_step[ticker]) < self.min_holding_period:
            return self.reward_invalid_trade

        current_price = self._get_current_price(ticker)
        if current_price <= 0 or np.isnan(current_price) or np.isinf(current_price):
            return self.reward_bad_price

        if action == 1:  # BUY
            max_invest = min(
                self.balance * self.buy_pct,
                self.equity * self.max_position_pct,
            )
            if max_invest < current_price * 1.1:
                return self.reward_invalid_trade

            execution_price = self._apply_slippage_buy(ticker, current_price)
            execution_price *= 1 + self.spread_bps

            quantity = max_invest / execution_price
            cost = quantity * execution_price
            total_cost = cost + cost * self.sec_fee + cost * self.finra_taf

            if total_cost <= self.balance:
                self.balance -= total_cost
                self.portfolio[ticker] += quantity
                self.entry_prices[ticker] = execution_price
                self.trades_today += 1
                self.total_trades += 1
                self.last_trade_step[ticker] = self.current_step
                return self.reward_buy_executed

        elif action == 2:  # SELL
            quantity = self.portfolio[ticker]
            if quantity < MIN_POSITION_THRESHOLD:
                return self.reward_invalid_trade

            execution_price = self._apply_slippage_sell(ticker, current_price)
            execution_price *= 1 - self.spread_bps

            proceeds = quantity * execution_price
            net_proceeds = proceeds - proceeds * self.sec_fee - proceeds * self.finra_taf

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

            if pnl > 0.01:
                return self.reward_trade_success
            return 0.0

        return 0.0

    def _calculate_reward(self, total_reward: float, trades_executed: int) -> float:
        """Calcule la récompense basée sur le Differential Sharpe Ratio (DSR)."""
        if len(self.portfolio_value_history) < 2:
            return 0.0

        prev_equity = self.portfolio_value_history[-2]

        ret = (self.equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
        self.returns_history.append(ret)

        return self.reward_calculator.calculate(
            prev_equity=prev_equity,
            current_equity=self.equity,
            peak_value=self.peak_value,
            trades_executed=trades_executed,
        )

    def _apply_slippage_buy(self, ticker: str, price: float) -> float:
        if self.slippage_model == "none":
            return price
        volume = self._get_current_volume(ticker)
        recent_prices = self._get_recent_prices(ticker)
        quantity = (self.balance * self.buy_pct) / price
        exec_price, _ = self.transaction_model.calculate_execution_price(
            ticker=ticker,
            intended_price=price,
            order_size=quantity,
            current_volume=volume,
            side="buy",
            recent_prices=recent_prices,
        )
        return exec_price

    def _apply_slippage_sell(self, ticker: str, price: float) -> float:
        if self.slippage_model == "none":
            return price
        volume = self._get_current_volume(ticker)
        recent_prices = self._get_recent_prices(ticker)
        quantity = self.portfolio[ticker]
        exec_price, _ = self.transaction_model.calculate_execution_price(
            ticker=ticker,
            intended_price=price,
            order_size=quantity,
            current_volume=volume,
            side="sell",
            recent_prices=recent_prices,
        )
        return exec_price

    def _get_current_price(self, ticker: str) -> float:
        prices = self.close_prices[ticker]
        if self.current_step >= len(prices):
            return float(prices[-1])
        price = float(prices[self.current_step])
        if np.isnan(price) or np.isinf(price) or price <= 0:
            return float(np.nanmedian(prices))
        return price

    def _get_current_volume(self, ticker: str) -> float:
        volumes = self.volume_arrays[ticker]
        if self.current_step >= len(volumes):
            return float(volumes[-1])
        return float(volumes[self.current_step])

    def _get_recent_prices(self, ticker: str) -> Optional[pd.Series]:
        prices = self.close_prices[ticker]
        start = max(0, self.current_step - 20)
        end = min(self.current_step + 1, len(prices))
        if end - start < 5:
            return None
        return pd.Series(prices[start:end])

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
        prices = {t: self._get_current_price(t) for t in self.tickers}
        return self.obs_builder.build(
            current_step=self.current_step,
            portfolio=self.portfolio,
            prices=prices,
            equity=self.equity,
            balance=self.balance,
            initial_balance=self.initial_balance,
            peak_value=self.peak_value,
        )

    def _get_info(self) -> dict:
        return {
            "equity": float(self.equity),
            "balance": float(self.balance),
            "total_return": float((self.equity - self.initial_balance) / self.initial_balance),
            "total_trades": int(self.total_trades),
            "winning_trades": int(self.winning_trades),
            "losing_trades": int(self.losing_trades),
            "current_step": int(self.current_step),
            "mode": self.mode,
        }

    def render(self, mode="human"):
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        print(
            f"Step: {self.current_step} | Equity: ${self.equity:,.2f} | "
            f"Return: {(self.equity / self.initial_balance - 1) * 100:.2f}% | "
            f"Trades: {self.total_trades} (WR: {win_rate:.1%})"
        )
