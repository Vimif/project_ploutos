# core/universal_environment_v8_lstm.py
"""Environnement V8 - Macro Data + Compatible LSTM (RecurrentPPO)

Améliorations V8 par rapport à V6:
- Intégration données macroéconomiques (VIX, TNX, DXY)
- Compatible RecurrentPPO (sb3-contrib) via observation 2D
- Mêmes features V2 (60+) + ~25 features macro = ~85+ features

L'environnement reste compatible avec PPO standard (MlpPolicy)
mais est optimisé pour RecurrentPPO (MlpLstmPolicy) qui ajoute
la mémoire temporelle.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Optional
from collections import deque

from core.advanced_features_v2 import AdvancedFeaturesV2
from core.macro_data import MacroDataFetcher
from core.transaction_costs import AdvancedTransactionModel

VALID_MODES = ("train", "eval", "backtest")


class UniversalTradingEnvV8LSTM(gym.Env):
    """Environnement V8 avec données macro et support LSTM.

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
        warmup_steps: int = 100,
        steps_per_trading_week: int = 78,
        drawdown_threshold: float = 0.10,
    ):
        super().__init__()

        if mode not in VALID_MODES:
            raise ValueError(f"mode doit être l'un de {VALID_MODES}, obtenu '{mode}'")
        self.mode = mode
        self.features_precomputed = features_precomputed  # Stocker le flag

        self._seed = seed
        self._rng = np.random.RandomState(seed)

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
        self.portfolio_value_history = deque(maxlen=252)
        self.returns_history = deque(maxlen=100)
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

        # DSR (Differential Sharpe Ratio) state - Init
        self.dsr_alpha = 0.05
        self.run_avg_ret = 0.0
        self.run_avg_sq_ret = 0.0

        # Préparer features techniques + macro
        self.macro_data = macro_data
        self._prepare_features(macro_data)

        # Observation space
        n_features_per_ticker = len(self.feature_columns)
        self.n_macro_features = len(self.macro_columns) if self.macro_columns else 0
        obs_size = (
            self.n_assets * n_features_per_ticker
            + self.n_macro_features
            + self.n_assets  # positions
            + 3  # cash_pct, total_return, drawdown
        )

        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(obs_size,), dtype=np.float32
        )

        self.action_space = gym.spaces.MultiDiscrete([3] * self.n_assets)

        mode_label = {"train": "Training", "eval": "Evaluation", "backtest": "Backtest"}
        print(
            f"Env V8 LSTM [{mode_label[self.mode]}]: "
            f"{self.n_assets} tickers x {n_features_per_ticker} features "
            f"+ {self.n_macro_features} macro = {obs_size} dims"
        )

    def _prepare_features(self, macro_data: Optional[pd.DataFrame]):
        """Préparer Features V2 + macro."""
        self.processed_data = {}
        self.feature_engineer = AdvancedFeaturesV2()
        self.macro_fetcher = MacroDataFetcher()

        for ticker in self.tickers:
            df = self.data[ticker].copy()  # Copie locale légère

            if not self.features_precomputed:
                # Calcul COÛTEUX (seulement si non pré-calculé)
                df = self.feature_engineer.calculate_all_features(df)

            self.processed_data[ticker] = df

        exclude_cols = ["Open", "High", "Low", "Close", "Volume"]
        self.feature_columns = [
            col for col in self.processed_data[self.tickers[0]].columns if col not in exclude_cols
        ]

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

        # Reset DSR state
        self.run_avg_ret = 0.0
        self.run_avg_sq_ret = 0.0

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
        reward = np.clip(reward, -10, 10)

        self.current_step += 1

        self.done = (
            self.current_step >= self.max_steps
            or self.equity < self.initial_balance * 0.5
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
            if quantity < 1e-6:
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
        current_equity = self.equity

        # Rendement pour ce step
        if prev_equity > 0:
            ret = (current_equity - prev_equity) / prev_equity
        else:
            ret = 0.0

        self.returns_history.append(ret)

        # --- Differential Sharpe Ratio (DSR) ---
        # Mise à jour des moyennes mobiles exponentielles (EMA)
        # A_t = A_{t-1} + alpha * (R_t - A_{t-1})
        # B_t = B_{t-1} + alpha * (R_t^2 - B_{t-1})

        delta_A = ret - self.run_avg_ret
        delta_B = (ret**2) - self.run_avg_sq_ret

        old_A = self.run_avg_ret
        old_B = self.run_avg_sq_ret

        self.run_avg_ret += self.dsr_alpha * delta_A
        self.run_avg_sq_ret += self.dsr_alpha * delta_B

        # Calcul du DSR (approximation directe de la dérivée)
        # D_t ~ (B_{t-1} * delta_A - 0.5 * A_{t-1} * delta_B) / (B_{t-1} - A_{t-1}^2)^(1.5)
        # Mais pour la stabilité numérique, on utilise une version simplifiée :
        # On récompense si le Sharpe augmente.

        variance = old_B - (old_A**2)
        if variance < 1e-6:  # Éviter division par zéro
            variance = 1e-6

        std_dev = np.sqrt(variance)

        # Formule de Moody & Saffell (2001)
        # D_t = (R_t - A_{t-1}) / std_{t-1}
        dsr = (ret - old_A) / std_dev

        # Reward Scale (pour que les valeurs soient ~ [-1, 1])
        reward = dsr * 0.1

        # --- Pénalités Additionnelles (Hybridation) ---

        # Drawdown Penalty (Critical)
        if self.use_drawdown_penalty and self.peak_value > 0:
            drawdown = (self.peak_value - current_equity) / self.peak_value
            if drawdown > self.drawdown_threshold:
                # Pénalité exponentielle
                reward -= drawdown * self.drawdown_penalty_factor * 0.5

        # Overtrading Penalty
        if trades_executed > 0:
            # Petit coût frictionnel pour éviter le "churning"
            reward -= self.penalty_overtrading * 0.1

        return reward * self.reward_scaling

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
        obs_parts = []

        # Features techniques par ticker
        for ticker in self.tickers:
            features_array = self.feature_arrays[ticker]
            if self.current_step >= len(features_array):
                features = np.zeros(len(self.feature_columns), dtype=np.float32)
            else:
                features = features_array[self.current_step]
            features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
            features = np.clip(features, -10, 10)
            obs_parts.append(features)

        # Features macro (partagées entre tous les tickers)
        if self.macro_array is not None:
            if self.current_step < len(self.macro_array):
                macro_features = self.macro_array[self.current_step]
            else:
                macro_features = np.zeros(len(self.macro_columns), dtype=np.float32)
            macro_features = np.nan_to_num(macro_features, nan=0.0, posinf=10.0, neginf=-10.0)
            macro_features = np.clip(macro_features, -10, 10)
            obs_parts.append(macro_features)

        # Positions
        for ticker in self.tickers:
            price = self._get_current_price(ticker)
            if price > 0:
                position_value = self.portfolio[ticker] * price
                position_pct = position_value / (self.equity + 1e-8)
            else:
                position_pct = 0.0
            obs_parts.append([np.clip(position_pct, 0, 1)])

        # Portfolio state
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
