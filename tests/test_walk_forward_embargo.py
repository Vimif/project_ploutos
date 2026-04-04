"""Walk-forward embargo wiring tests."""

import pandas as pd

from training.train import generate_walk_forward_splits, run_walk_forward


def _sample_market_data() -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2020-01-01", "2024-01-01", freq="h")
    df = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.5,
            "Volume": 1000,
        },
        index=dates,
    )
    return {"TEST": df}


def test_generate_walk_forward_splits_applies_embargo_months():
    data = _sample_market_data()

    splits = generate_walk_forward_splits(
        data,
        train_years=1,
        test_months=3,
        step_months=3,
        embargo_months=2,
    )

    assert splits, "Expected at least one walk-forward split"
    first_split = splits[0]
    expected_test_start = (pd.Timestamp(first_split["train_end"]) + pd.DateOffset(months=2)).date()
    assert pd.Timestamp(first_split["test_start"]).date() == expected_test_start


def test_run_walk_forward_passes_embargo_months_from_config(monkeypatch):
    captured = {}
    config = {
        "data": {"tickers": ["TEST"], "period": "2y", "interval": "1h"},
        "training": {"n_envs": 1},
        "environment": {"initial_balance": 10_000, "commission": 0.001},
        "walk_forward": {
            "train_years": 1,
            "test_months": 3,
            "step_months": 3,
            "embargo_months": 4,
        },
    }

    class _MacroFetcher:
        def fetch_all(self, *args, **kwargs):
            return pd.DataFrame()

    def _fake_generate_walk_forward_splits(*args, **kwargs):
        captured["embargo_months"] = kwargs["embargo_months"]
        return []

    monkeypatch.setattr("training.train.load_config", lambda _path: config)
    monkeypatch.setattr("training.train.download_data", lambda **_kwargs: _sample_market_data())
    monkeypatch.setattr("training.train.MacroDataFetcher", lambda: _MacroFetcher())
    monkeypatch.setattr(
        "training.train.generate_walk_forward_splits",
        _fake_generate_walk_forward_splits,
    )

    run_walk_forward("config/config.yaml")

    assert captured["embargo_months"] == 4
