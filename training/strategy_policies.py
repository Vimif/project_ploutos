"""Strategy policy abstractions for model family bake-offs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

SUPPORTED_STRATEGY_FAMILIES = (
    "ppo_single",
    "ppo_ensemble",
    "recurrent_ppo",
    "supervised_ranker",
    "rule_momentum_regime",
)
OHLCV_COLUMNS = {"Open", "High", "Low", "Close", "Volume", "Date", "Datetime", "Timestamp"}


@dataclass
class StrategyContext:
    """Unified inference context for all strategy families."""

    observation: np.ndarray
    current_step: int
    tickers: list[str]
    prices: dict[str, float]
    portfolio: dict[str, float]
    entry_prices: dict[str, float]
    processed_data: dict[str, pd.DataFrame]
    feature_columns: list[str]
    macro_columns: list[str]
    macro_row: Optional[dict[str, float]]
    interval: str
    regime_risk_on: bool


class StrategyPolicy(ABC):
    """Common interface for all candidate strategy families."""

    family: str

    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.seed = int(seed)

    @abstractmethod
    def fit(
        self,
        train_data: dict[str, pd.DataFrame],
        macro_data: Optional[pd.DataFrame],
        artifact_dir: Optional[str] = None,
    ) -> "StrategyPolicy":
        """Train or initialize the strategy for one fold."""

    @abstractmethod
    def predict_actions(self, context: StrategyContext) -> np.ndarray:
        """Return one MultiDiscrete action vector for the current step."""

    @abstractmethod
    def reset_state(self) -> None:
        """Reset any recurrent or per-episode inference state."""

    @abstractmethod
    def artifact_metadata(self) -> dict:
        """Return model-specific metadata for reporting."""

    def close(self) -> None:
        """Release optional resources after one fold."""


def _strategy_cfg(config: dict) -> dict:
    return dict(config.get("strategy", {}))


def _training_cfg(config: dict) -> dict:
    return dict(config.get("training", {}))


def _network_cfg(config: dict) -> dict:
    return dict(config.get("network", {}))


def _live_cfg(config: dict) -> dict:
    return dict(config.get("live", {}))


def _select_feature_columns(
    data: dict[str, pd.DataFrame], max_features_per_ticker: int
) -> list[str]:
    ref_ticker = next(iter(data))
    ref_df = data[ref_ticker]
    feature_columns = [
        col
        for col in ref_df.columns
        if col not in OHLCV_COLUMNS and pd.api.types.is_numeric_dtype(ref_df[col])
    ]
    if max_features_per_ticker > 0 and len(feature_columns) > max_features_per_ticker:
        variances = ref_df[feature_columns].var().fillna(0)
        feature_columns = variances.nlargest(max_features_per_ticker).index.tolist()
    return feature_columns


def _align_macro(macro_data: Optional[pd.DataFrame], index: pd.Index) -> pd.DataFrame:
    if macro_data is None or macro_data.empty:
        return pd.DataFrame(index=index)
    aligned = macro_data.reindex(index).ffill().bfill()
    return aligned.fillna(0.0)


def _estimate_round_trip_cost(config: dict) -> float:
    env_cfg = config.get("environment", {})
    commission = float(env_cfg.get("commission", 0.0))
    sec_fee = float(env_cfg.get("sec_fee", 0.0))
    finra_taf = float(env_cfg.get("finra_taf", 0.0))
    spread_bps = float(env_cfg.get("spread_bps", 0.0)) / 10000
    market_impact_factor = float(env_cfg.get("market_impact_factor", 0.0))
    return float((2 * spread_bps) + commission + sec_fee + finra_taf + market_impact_factor)


class _SB3StrategyBase(StrategyPolicy):
    """Common SB3-backed strategy wrapper."""

    model_class_name = "PPO"
    is_recurrent = False

    def __init__(self, config: dict, seed: int = 42):
        super().__init__(config, seed=seed)
        self.models: list = []
        self.vecnorms: list = []
        self._recurrent_states: list = []

    def _runtime_imports(self):
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

        try:
            from sb3_contrib import RecurrentPPO
        except ImportError:
            RecurrentPPO = None

        from core.environment import TradingEnv

        return {
            "torch": torch,
            "PPO": PPO,
            "Monitor": Monitor,
            "DummyVecEnv": DummyVecEnv,
            "SubprocVecEnv": SubprocVecEnv,
            "VecNormalize": VecNormalize,
            "RecurrentPPO": RecurrentPPO,
            "TradingEnv": TradingEnv,
        }

    def _build_env(self, train_data, macro_data, runtime):
        training_cfg = _training_cfg(self.config)
        env_kwargs = {k: v for k, v in self.config.get("environment", {}).items()}
        env_kwargs["mode"] = "train"
        env_kwargs["features_precomputed"] = True
        n_envs = int(training_cfg.get("n_envs", 1))

        def make_env():
            return runtime["Monitor"](
                runtime["TradingEnv"](train_data, macro_data=macro_data, **env_kwargs)
            )

        if self.is_recurrent:
            env = runtime["DummyVecEnv"]([make_env for _ in range(n_envs)])
        else:
            env = runtime["SubprocVecEnv"]([make_env for _ in range(n_envs)])
        return runtime["VecNormalize"](
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=float(training_cfg.get("gamma", 0.99)),
        )

    def _build_model(self, env, runtime, seed: int):
        training_cfg = _training_cfg(self.config)
        network_cfg = _network_cfg(self.config)
        net_arch = network_cfg.get("net_arch", [256, 128])
        activation_name = network_cfg.get("activation_fn", "tanh")
        activation_fn = (
            runtime["torch"].nn.Tanh if activation_name == "tanh" else runtime["torch"].nn.ReLU
        )
        device = "cuda" if runtime["torch"].cuda.is_available() else "cpu"

        if self.is_recurrent:
            recurrent_cls = runtime["RecurrentPPO"]
            if recurrent_cls is None:
                raise RuntimeError("sb3-contrib is required for recurrent_ppo")
            policy_kwargs = {
                "net_arch": net_arch,
                "activation_fn": activation_fn,
                "lstm_hidden_size": int(network_cfg.get("lstm_hidden_size", 256)),
                "n_lstm_layers": int(network_cfg.get("n_lstm_layers", 1)),
            }
            return recurrent_cls(
                "MlpLstmPolicy",
                env,
                learning_rate=float(training_cfg.get("learning_rate", 0.0003)),
                n_steps=int(training_cfg.get("n_steps", 512)),
                batch_size=int(training_cfg.get("batch_size", 128)),
                n_epochs=int(training_cfg.get("n_epochs", 10)),
                gamma=float(training_cfg.get("gamma", 0.99)),
                gae_lambda=float(training_cfg.get("gae_lambda", 0.95)),
                clip_range=float(training_cfg.get("clip_range", 0.2)),
                ent_coef=float(training_cfg.get("ent_coef", 0.01)),
                vf_coef=float(training_cfg.get("vf_coef", 0.5)),
                max_grad_norm=float(training_cfg.get("max_grad_norm", 0.5)),
                policy_kwargs=policy_kwargs,
                verbose=0,
                device=device,
                seed=seed,
            )

        policy_kwargs = {
            "net_arch": [{"pi": net_arch, "vf": net_arch}],
            "activation_fn": activation_fn,
        }
        return runtime["PPO"](
            "MlpPolicy",
            env,
            learning_rate=float(training_cfg.get("learning_rate", 0.0003)),
            n_steps=int(training_cfg.get("n_steps", 512)),
            batch_size=int(training_cfg.get("batch_size", 128)),
            n_epochs=int(training_cfg.get("n_epochs", 10)),
            gamma=float(training_cfg.get("gamma", 0.99)),
            gae_lambda=float(training_cfg.get("gae_lambda", 0.95)),
            clip_range=float(training_cfg.get("clip_range", 0.2)),
            ent_coef=float(training_cfg.get("ent_coef", 0.01)),
            vf_coef=float(training_cfg.get("vf_coef", 0.5)),
            max_grad_norm=float(training_cfg.get("max_grad_norm", 0.5)),
            target_kl=float(training_cfg.get("target_kl", 0.02)),
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=device,
            seed=seed,
        )

    def _n_models(self) -> int:
        return 1

    def fit(
        self,
        train_data: dict[str, pd.DataFrame],
        macro_data: Optional[pd.DataFrame],
        artifact_dir: Optional[str] = None,
    ) -> "_SB3StrategyBase":
        del artifact_dir

        runtime = self._runtime_imports()
        timesteps = int(_training_cfg(self.config).get("total_timesteps", 100_000))
        self.models = []
        self.vecnorms = []
        self._recurrent_states = []

        for model_idx in range(self._n_models()):
            env = self._build_env(train_data, macro_data, runtime)
            model_seed = self.seed + model_idx * 1000
            model = self._build_model(env, runtime, model_seed)
            model.learn(total_timesteps=timesteps, progress_bar=False)
            env.training = False
            env.norm_reward = False
            self.models.append(model)
            self.vecnorms.append(env)
            self._recurrent_states.append(None)
        return self

    def predict_actions(self, context: StrategyContext) -> np.ndarray:
        if not self.models:
            raise RuntimeError(f"{self.family} must be fitted before predict_actions")

        observation = np.asarray(context.observation, dtype=np.float32)
        min_confidence = float(_live_cfg(self.config).get("min_confidence", 0.5))
        all_actions = []
        for idx, (model, vecnorm) in enumerate(zip(self.models, self.vecnorms)):
            obs = vecnorm.normalize_obs(observation.reshape(1, -1)).flatten()
            if self.is_recurrent:
                episode_start = np.array([self._recurrent_states[idx] is None], dtype=bool)
                action, state = model.predict(
                    obs,
                    state=self._recurrent_states[idx],
                    episode_start=episode_start,
                    deterministic=True,
                )
                self._recurrent_states[idx] = state
            else:
                action, _ = model.predict(obs, deterministic=True)
            all_actions.append(np.asarray(action, dtype=np.int64))

        if len(all_actions) == 1:
            return np.asarray(all_actions[0], dtype=np.int64).flatten()

        stacked = np.asarray(all_actions, dtype=np.int64)
        final_actions = np.zeros(stacked.shape[1], dtype=np.int64)
        for asset_idx in range(stacked.shape[1]):
            values, counts = np.unique(stacked[:, asset_idx], return_counts=True)
            best_idx = int(np.argmax(counts))
            confidence = float(counts[best_idx] / len(all_actions))
            selected_action = int(values[best_idx])
            if selected_action == 1 and confidence < min_confidence:
                selected_action = 0
            final_actions[asset_idx] = selected_action
        return final_actions

    def reset_state(self) -> None:
        self._recurrent_states = [None for _ in self.models]

    def artifact_metadata(self) -> dict:
        return {
            "family": self.family,
            "model_class": self.model_class_name,
            "n_models": len(self.models),
            "seed": self.seed,
        }

    def close(self) -> None:
        for vecnorm in self.vecnorms:
            close = getattr(vecnorm, "close", None)
            if callable(close):
                close()


class PPOSingleStrategyPolicy(_SB3StrategyBase):
    family = "ppo_single"
    model_class_name = "PPO"


class PPOEnsembleStrategyPolicy(_SB3StrategyBase):
    family = "ppo_ensemble"
    model_class_name = "PPO"

    def _n_models(self) -> int:
        live_cfg = _live_cfg(self.config)
        strategy_cfg = _strategy_cfg(self.config)
        return int(strategy_cfg.get("ensemble_size", live_cfg.get("ensemble_size", 3)))


class RecurrentPPOStrategyPolicy(_SB3StrategyBase):
    family = "recurrent_ppo"
    model_class_name = "RecurrentPPO"
    is_recurrent = True


class SupervisedRankerStrategyPolicy(StrategyPolicy):
    family = "supervised_ranker"

    def __init__(self, config: dict, seed: int = 42):
        super().__init__(config, seed=seed)
        self.models: dict[str, object] = {}
        self.feature_columns: list[str] = []
        self.macro_columns: list[str] = []

    def fit(
        self,
        train_data: dict[str, pd.DataFrame],
        macro_data: Optional[pd.DataFrame],
        artifact_dir: Optional[str] = None,
    ) -> "SupervisedRankerStrategyPolicy":
        del artifact_dir

        from sklearn.ensemble import HistGradientBoostingRegressor

        strategy_cfg = _strategy_cfg(self.config)
        max_features = int(self.config.get("environment", {}).get("max_features_per_ticker", 0))
        forward_bars = int(strategy_cfg.get("supervised_forward_bars", 4))
        self.feature_columns = _select_feature_columns(train_data, max_features)
        self.models = {}
        self.macro_columns = (
            list(macro_data.columns) if macro_data is not None and not macro_data.empty else []
        )
        cost = _estimate_round_trip_cost(self.config)

        for ticker, df in train_data.items():
            aligned_macro = _align_macro(macro_data, df.index)
            feature_frame = df[self.feature_columns].copy()
            if not aligned_macro.empty:
                feature_frame = feature_frame.join(aligned_macro, how="left")
            feature_frame = feature_frame.fillna(0.0)

            future_return = (df["Close"].shift(-forward_bars) / df["Close"]) - 1.0 - cost
            train_frame = feature_frame.copy()
            train_frame["target"] = future_return
            train_frame = train_frame.replace([np.inf, -np.inf], 0.0).dropna()
            if len(train_frame) < 20:
                continue

            model = HistGradientBoostingRegressor(
                max_depth=3,
                learning_rate=0.05,
                max_iter=100,
                random_state=self.seed,
            )
            model.fit(train_frame.drop(columns=["target"]).values, train_frame["target"].values)
            self.models[ticker] = model

        if not self.models:
            raise RuntimeError("supervised_ranker could not fit any ticker model")
        return self

    def predict_actions(self, context: StrategyContext) -> np.ndarray:
        strategy_cfg = _strategy_cfg(self.config)
        buy_threshold = float(strategy_cfg.get("supervised_buy_threshold", 0.002))
        sell_threshold = float(strategy_cfg.get("supervised_sell_threshold", 0.0))

        actions = np.zeros(len(context.tickers), dtype=np.int64)
        scores: list[tuple[int, float]] = []
        for idx, ticker in enumerate(context.tickers):
            model = self.models.get(ticker)
            if model is None:
                continue
            row = context.processed_data[ticker].iloc[context.current_step]
            features = row[self.feature_columns].astype(float).fillna(0.0).to_dict()
            if context.macro_row:
                features.update(context.macro_row)
            ordered = np.array(
                [features.get(col, 0.0) for col in list(self.feature_columns) + self.macro_columns]
            )
            score = float(model.predict(ordered.reshape(1, -1))[0])

            if context.portfolio.get(ticker, 0.0) > 0 and score < sell_threshold:
                actions[idx] = 2
                continue
            if (
                context.portfolio.get(ticker, 0.0) <= 0
                and score > buy_threshold
                and context.regime_risk_on
            ):
                scores.append((idx, score))

        if scores:
            best_idx = max(scores, key=lambda item: item[1])[0]
            actions[best_idx] = 1
        return actions

    def reset_state(self) -> None:
        return None

    def artifact_metadata(self) -> dict:
        return {
            "family": self.family,
            "seed": self.seed,
            "n_models": len(self.models),
            "feature_count": len(self.feature_columns),
            "forward_bars": int(_strategy_cfg(self.config).get("supervised_forward_bars", 4)),
        }


class RuleMomentumRegimeStrategyPolicy(StrategyPolicy):
    family = "rule_momentum_regime"

    def __init__(self, config: dict, seed: int = 42):
        super().__init__(config, seed=seed)
        strategy_cfg = _strategy_cfg(config)
        self.fast_ma = int(strategy_cfg.get("rule_fast_ma", 20))
        self.slow_ma = int(strategy_cfg.get("rule_slow_ma", 50))
        self.lookback = int(strategy_cfg.get("rule_momentum_lookback", 4))

    def fit(
        self,
        train_data: dict[str, pd.DataFrame],
        macro_data: Optional[pd.DataFrame],
        artifact_dir: Optional[str] = None,
    ) -> "RuleMomentumRegimeStrategyPolicy":
        del train_data, macro_data, artifact_dir
        return self

    def predict_actions(self, context: StrategyContext) -> np.ndarray:
        actions = np.zeros(len(context.tickers), dtype=np.int64)
        scores: list[tuple[int, float]] = []
        for idx, ticker in enumerate(context.tickers):
            close = context.processed_data[ticker]["Close"].iloc[: context.current_step + 1]
            if len(close) < self.slow_ma:
                continue
            fast_ma = float(close.tail(self.fast_ma).mean())
            slow_ma = float(close.tail(self.slow_ma).mean())
            momentum = 0.0
            if len(close) > self.lookback:
                previous = float(close.iloc[-self.lookback - 1])
                if previous > 0:
                    momentum = (float(close.iloc[-1]) / previous) - 1.0

            held = context.portfolio.get(ticker, 0.0) > 0
            score = ((fast_ma - slow_ma) / max(slow_ma, 1e-8)) + momentum
            if held and (fast_ma < slow_ma or momentum < 0.0):
                actions[idx] = 2
            elif not held and score > 0 and context.regime_risk_on:
                scores.append((idx, score))

        if scores:
            best_idx = max(scores, key=lambda item: item[1])[0]
            actions[best_idx] = 1
        return actions

    def reset_state(self) -> None:
        return None

    def artifact_metadata(self) -> dict:
        return {
            "family": self.family,
            "fast_ma": self.fast_ma,
            "slow_ma": self.slow_ma,
            "lookback": self.lookback,
        }


def build_strategy_policy(family: str, config: dict, seed: int = 42) -> StrategyPolicy:
    """Instantiate one supported strategy family."""

    normalized = family.strip().lower()
    if normalized == "ppo_single":
        return PPOSingleStrategyPolicy(config, seed=seed)
    if normalized == "ppo_ensemble":
        return PPOEnsembleStrategyPolicy(config, seed=seed)
    if normalized == "recurrent_ppo":
        return RecurrentPPOStrategyPolicy(config, seed=seed)
    if normalized == "supervised_ranker":
        return SupervisedRankerStrategyPolicy(config, seed=seed)
    if normalized == "rule_momentum_regime":
        return RuleMomentumRegimeStrategyPolicy(config, seed=seed)
    raise ValueError(f"Unsupported strategy family: {family}")
