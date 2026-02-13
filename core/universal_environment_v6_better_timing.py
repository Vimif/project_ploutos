# core/universal_environment_v6_better_timing.py
"""Environnement V6 - MEILLEUR TIMING avec Features V2

Objectif: Résoudre le problème "buy high" (85% mauvais timing)

Améliorations V6:
- Features V2 optimisées pour détecter bons points d'entrée
- 60+ features par ticker (vs 37 avant)
- Entry score composite
- Support/Resistance dynamiques

Améliorations Étage 1 (refactoring):
- Modes train/eval/backtest séparés
- Intégration AdvancedTransactionModel pour backtest réaliste
- Reward params configurables (plus de magic numbers)
- Reproductibilité via seed en mode backtest
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Optional
from collections import deque

from core.advanced_features_v2 import AdvancedFeaturesV2
from core.transaction_costs import AdvancedTransactionModel

# Modes valides pour l'environnement
VALID_MODES = ("train", "eval", "backtest")


class UniversalTradingEnvV6BetterTiming(gym.Env):
    """Environnement avec Features V2 pour meilleur timing.

    Modes:
        train:    Random start, slippage stochastique, léger noise. Pour PPO.
        eval:     Start fixe (step 100), slippage moyen fixe. Pour EvalCallback.
        backtest: Start 0, AdvancedTransactionModel, seed fixé, reproductible.
    """

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
        reward_scaling: float = 1.5,
        use_sharpe_penalty: bool = True,
        use_drawdown_penalty: bool = True,
        max_trades_per_day: int = 10,
        min_holding_period: int = 2,
        reward_trade_success: float = 0.5,
        penalty_overtrading: float = 0.005,
        drawdown_penalty_factor: float = 3.0,
        # ===== Nouveaux paramètres Étage 1 =====
        mode: str = "train",
        seed: Optional[int] = None,
        # Rewards configurables (ex-magic numbers)
        reward_buy_executed: float = 0.1,
        reward_overtrading: float = -0.02,
        reward_invalid_trade: float = -0.01,
        reward_bad_price: float = -0.05,
        reward_good_return_bonus: float = 0.3,
        reward_high_winrate_bonus: float = 0.2,
        good_return_threshold: float = 0.01,
        high_winrate_threshold: float = 0.6,
    ):
        super().__init__()

        # ===== Mode =====
        if mode not in VALID_MODES:
            raise ValueError(
                f"mode doit être l'un de {VALID_MODES}, obtenu '{mode}'"
            )
        self.mode = mode

        # ===== Seed / Reproductibilité =====
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

        # ===== Rewards configurables =====
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

        # ===== AdvancedTransactionModel (pour mode backtest) =====
        self.transaction_model = AdvancedTransactionModel(
            base_commission=commission,
            market_impact_coef=market_impact_factor,
            rng=self._rng,
        )

        # ✅ Features V2
        self._prepare_features_v2()

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

        mode_label = {"train": "Training", "eval": "Evaluation", "backtest": "Backtest"}
        print(
            f"✅ Env V6 [{mode_label[self.mode]}]: "
            f"{self.n_assets} tickers × {n_features_per_ticker} features "
            f"= {obs_size} dims"
        )

    def _prepare_features_v2(self):
        """Préparer Features V2 optimisées."""
        print(f"  🚀 Calcul Features V2 (optimisées pour timing)...")

        self.processed_data = {}
        self.feature_engineer = AdvancedFeaturesV2()

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
        print(f"      Include: entry_score, support/resistance, divergences, etc.")

        # ⚡ OPTIMISATION: Convertir en numpy arrays pour accès rapide
        self.feature_arrays = {}
        self.close_prices = {}
        self.volume_arrays = {}

        for ticker in self.tickers:
            df = self.processed_data[ticker]
            # ⚡ OPTIMISATION: Pré-calculer nan_to_num et clip ici (une seule fois)
            features = df[self.feature_columns].values.astype(np.float32)
            features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
            features = np.clip(features, -10.0, 10.0)
            self.feature_arrays[ticker] = features

            self.close_prices[ticker] = df['Close'].values.astype(np.float32)
            # Volume pour AdvancedTransactionModel
            if 'Volume' in df.columns:
                self.volume_arrays[ticker] = df['Volume'].values.astype(np.float64)
            else:
                self.volume_arrays[ticker] = np.full(len(df), 1_000_000.0)

        self.max_steps = min(
            self.max_steps,
            min(len(df) for df in self.processed_data.values()) - 100
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Utiliser le seed du mode si fourni
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

        # Start step dépend du mode
        if self.mode == "train":
            self.current_step = self._rng.randint(100, max(101, self.max_steps // 2))
        elif self.mode == "eval":
            self.current_step = 100
        elif self.mode == "backtest":
            self.current_step = 0

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
            self.current_step >= self.max_steps or
            self.equity < self.initial_balance * 0.5 or
            self.balance < 0
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
                self.equity * self.max_position_pct
            )

            if max_invest < current_price * 1.1:
                return self.reward_invalid_trade

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
            execution_price *= (1 - self.spread_bps)

            proceeds = quantity * execution_price

            sec_fee_cost = proceeds * self.sec_fee
            finra_fee_cost = proceeds * self.finra_taf
            net_proceeds = proceeds - sec_fee_cost - finra_fee_cost

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

    def _calculate_reward(self, trade_reward: float, trades_executed: int) -> float:
        if len(self.portfolio_value_history) < 2:
            return 0.0

        prev_equity = self.portfolio_value_history[-2]
        current_equity = self.equity

        if prev_equity <= 0:
            return 0.0

        pct_return = (current_equity - prev_equity) / prev_equity
        self.returns_history.append(pct_return)

        reward = pct_return * 100

        if self.use_drawdown_penalty and self.peak_value > 0:
            drawdown = (self.peak_value - current_equity) / self.peak_value
            if drawdown > 0.15:
                reward -= drawdown * self.drawdown_penalty_factor

        if trades_executed > 2:
            reward -= self.penalty_overtrading * trades_executed

        if pct_return > self.good_return_threshold:
            reward += self.reward_good_return_bonus

        if self.total_trades > 10:
            win_rate = self.winning_trades / self.total_trades
            if win_rate > self.high_winrate_threshold:
                reward += self.reward_high_winrate_bonus

        return reward * self.reward_scaling

    def _apply_slippage_buy(self, ticker: str, price: float) -> float:
        """Applique le slippage à l'achat selon le mode."""
        if self.slippage_model == 'none':
            return price

        if self.mode == "backtest":
            # Mode backtest: utiliser AdvancedTransactionModel
            volume = self._get_current_volume(ticker)
            recent_prices = self._get_recent_prices(ticker)
            quantity = (self.balance * self.buy_pct) / price
            exec_price, _ = self.transaction_model.calculate_execution_price(
                ticker=ticker,
                intended_price=price,
                order_size=quantity,
                current_volume=volume,
                side='buy',
                recent_prices=recent_prices,
            )
            return exec_price
        else:
            # Mode train/eval: slippage stochastique
            slippage_pct = self._rng.uniform(0.0001, 0.001)
            return price * (1 + slippage_pct)

    def _apply_slippage_sell(self, ticker: str, price: float) -> float:
        """Applique le slippage à la vente selon le mode."""
        if self.slippage_model == 'none':
            return price

        if self.mode == "backtest":
            # Mode backtest: utiliser AdvancedTransactionModel
            volume = self._get_current_volume(ticker)
            recent_prices = self._get_recent_prices(ticker)
            quantity = self.portfolio[ticker]
            exec_price, _ = self.transaction_model.calculate_execution_price(
                ticker=ticker,
                intended_price=price,
                order_size=quantity,
                current_volume=volume,
                side='sell',
                recent_prices=recent_prices,
            )
            return exec_price
        else:
            # Mode train/eval: slippage stochastique
            slippage_pct = self._rng.uniform(0.0001, 0.001)
            return price * (1 - slippage_pct)

    def _get_current_price(self, ticker: str) -> float:
        prices = self.close_prices[ticker]
        if self.current_step >= len(prices):
            return float(prices[-1])

        price = float(prices[self.current_step])

        if np.isnan(price) or np.isinf(price) or price <= 0:
            # Fallback (rare)
            return float(np.nanmedian(prices))

        return price

    def _get_current_volume(self, ticker: str) -> float:
        """Retourne le volume actuel pour le ticker."""
        volumes = self.volume_arrays[ticker]
        if self.current_step >= len(volumes):
            return float(volumes[-1])
        return float(volumes[self.current_step])

    def _get_recent_prices(self, ticker: str) -> Optional[pd.Series]:
        """Retourne les 20 derniers prix pour le calcul de volatilité."""
        prices = self.close_prices[ticker]
        start = max(0, self.current_step - 20)
        end = self.current_step + 1
        if end > len(prices):
            end = len(prices)
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

        for ticker in self.tickers:
            features_array = self.feature_arrays[ticker]

            if self.current_step >= len(features_array):
                features = np.zeros(len(self.feature_columns), dtype=np.float32)
            else:
                features = features_array[self.current_step]

            # ⚡ OPTIMISATION: Déjà fait dans _prepare_features_v2
            # features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
            # features = np.clip(features, -10, 10)

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
        total_return = np.clip(
            (self.equity - self.initial_balance) / self.initial_balance, -1, 5
        )
        drawdown = np.clip(
            (self.peak_value - self.equity) / (self.peak_value + 1e-8), 0, 1
        )

        obs_parts.append([cash_pct, total_return, drawdown])

        obs = np.concatenate([np.array(p).flatten() for p in obs_parts])

        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = np.clip(obs, -10, 10)

        return obs.astype(np.float32)

    def _get_info(self) -> dict:
        return {
            'equity': float(self.equity),
            'balance': float(self.balance),
            'total_return': float(
                (self.equity - self.initial_balance) / self.initial_balance
            ),
            'total_trades': int(self.total_trades),
            'winning_trades': int(self.winning_trades),
            'losing_trades': int(self.losing_trades),
            'current_step': int(self.current_step),
            'mode': self.mode,
        }

    def render(self, mode='human'):
        win_rate = (
            self.winning_trades / self.total_trades
            if self.total_trades > 0
            else 0
        )
        print(
            f"Step: {self.current_step} | Equity: ${self.equity:,.2f} | "
            f"Return: {(self.equity / self.initial_balance - 1) * 100:.2f}% | "
            f"Trades: {self.total_trades} (WR: {win_rate:.1%})"
        )
