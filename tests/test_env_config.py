"""Tests for EnvConfig dataclass and from_flat_dict."""

import pytest
from core.env_config import EnvConfig, TransactionConfig, RewardConfig, TradingConfig


class TestDefaults:
    def test_default_config(self):
        cfg = EnvConfig()
        assert cfg.initial_balance == 100_000.0
        assert cfg.max_features_per_ticker == 0
        assert cfg.features_precomputed is False

    def test_default_transaction(self):
        cfg = EnvConfig()
        assert cfg.transaction.commission == 0.0
        assert cfg.transaction.spread_bps == 5.0
        assert cfg.transaction.slippage_model == "realistic"

    def test_default_reward(self):
        cfg = EnvConfig()
        assert cfg.reward.reward_scaling == 1.5
        assert cfg.reward.use_drawdown_penalty is True

    def test_default_trading(self):
        cfg = EnvConfig()
        assert cfg.trading.max_steps == 2500
        assert cfg.trading.max_trades_per_day == 10
        assert cfg.trading.min_holding_period == 2


class TestFromFlatDict:
    def test_empty_dict(self):
        cfg = EnvConfig.from_flat_dict({})
        assert cfg.initial_balance == 100_000.0

    def test_transaction_fields(self):
        cfg = EnvConfig.from_flat_dict({"commission": 0.01, "spread_bps": 10.0})
        assert cfg.transaction.commission == 0.01
        assert cfg.transaction.spread_bps == 10.0

    def test_reward_fields(self):
        cfg = EnvConfig.from_flat_dict({"reward_scaling": 2.0, "use_drawdown_penalty": False})
        assert cfg.reward.reward_scaling == 2.0
        assert cfg.reward.use_drawdown_penalty is False

    def test_trading_fields(self):
        cfg = EnvConfig.from_flat_dict({"max_steps": 1000, "buy_pct": 0.10})
        assert cfg.trading.max_steps == 1000
        assert cfg.trading.buy_pct == 0.10

    def test_top_level_fields(self):
        cfg = EnvConfig.from_flat_dict(
            {"initial_balance": 50000, "max_features_per_ticker": 20}
        )
        assert cfg.initial_balance == 50000
        assert cfg.max_features_per_ticker == 20

    def test_unknown_keys_ignored(self):
        cfg = EnvConfig.from_flat_dict({"nonexistent_key": 999})
        assert cfg.initial_balance == 100_000.0

    def test_mixed_fields(self):
        cfg = EnvConfig.from_flat_dict({
            "initial_balance": 200000,
            "commission": 0.005,
            "reward_scaling": 3.0,
            "max_steps": 500,
            "features_precomputed": True,
        })
        assert cfg.initial_balance == 200000
        assert cfg.transaction.commission == 0.005
        assert cfg.reward.reward_scaling == 3.0
        assert cfg.trading.max_steps == 500
        assert cfg.features_precomputed is True
