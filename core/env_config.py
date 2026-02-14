# core/env_config.py
"""Dataclass configuration for TradingEnv.

Replaces the 30+ constructor parameters with structured, typed config objects.
"""

from dataclasses import dataclass, field


@dataclass
class TransactionConfig:
    """Transaction cost parameters."""

    commission: float = 0.0
    sec_fee: float = 0.0000221
    finra_taf: float = 0.000145
    slippage_model: str = "realistic"
    spread_bps: float = 5.0
    market_impact_factor: float = 0.00015


@dataclass
class RewardConfig:
    """Reward function parameters."""

    reward_scaling: float = 1.5
    use_sharpe_penalty: bool = True
    use_drawdown_penalty: bool = True
    reward_trade_success: float = 0.5
    penalty_overtrading: float = 0.005
    drawdown_penalty_factor: float = 3.0
    reward_buy_executed: float = 0.1
    reward_overtrading: float = -0.02
    reward_invalid_trade: float = -0.01
    reward_bad_price: float = -0.05
    reward_good_return_bonus: float = 0.3
    reward_high_winrate_bonus: float = 0.2
    good_return_threshold: float = 0.01
    high_winrate_threshold: float = 0.6


@dataclass
class TradingConfig:
    """Trading rules and constraints."""

    max_steps: int = 2500
    buy_pct: float = 0.20
    max_position_pct: float = 0.25
    max_trades_per_day: int = 10
    min_holding_period: int = 2
    warmup_steps: int = 100
    steps_per_trading_week: int = 78
    drawdown_threshold: float = 0.10


@dataclass
class EnvConfig:
    """Top-level environment configuration."""

    initial_balance: float = 100_000.0
    transaction: TransactionConfig = field(default_factory=TransactionConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    max_features_per_ticker: int = 0
    features_precomputed: bool = False

    @classmethod
    def from_flat_dict(cls, d: dict) -> "EnvConfig":
        """Build EnvConfig from a flat dict (e.g. YAML environment section).

        Maps flat keys like 'commission', 'reward_scaling', 'max_steps'
        to the correct nested dataclass.
        """
        tx_fields = {f.name for f in TransactionConfig.__dataclass_fields__.values()}
        rw_fields = {f.name for f in RewardConfig.__dataclass_fields__.values()}
        tr_fields = {f.name for f in TradingConfig.__dataclass_fields__.values()}
        top_fields = {"initial_balance", "max_features_per_ticker", "features_precomputed"}

        tx_kwargs = {}
        rw_kwargs = {}
        tr_kwargs = {}
        top_kwargs = {}

        for k, v in d.items():
            if k in tx_fields:
                tx_kwargs[k] = v
            elif k in rw_fields:
                rw_kwargs[k] = v
            elif k in tr_fields:
                tr_kwargs[k] = v
            elif k in top_fields:
                top_kwargs[k] = v
            # Unknown keys are silently ignored (validated elsewhere by schema.py)

        return cls(
            transaction=TransactionConfig(**tx_kwargs),
            reward=RewardConfig(**rw_kwargs),
            trading=TradingConfig(**tr_kwargs),
            **top_kwargs,
        )
