import sys
from unittest.mock import MagicMock

if "torch" not in sys.modules:
    sys.modules["torch"] = MagicMock()

import pytest
import numpy as np
import pandas as pd
from core.data_pipeline import DataSplitter, DataSplit


def _make_fake_data(n_tickers: int = 3, n_bars: int = 1000) -> dict:
    dates = pd.date_range("2023-01-01", periods=n_bars, freq="D")
    data = {}
    for i in range(n_tickers):
        ticker = f"TICK{i}"
        prices = 100 + np.random.randn(n_bars).cumsum()
        data[ticker] = pd.DataFrame(
            {
                "Open": prices + np.random.rand(n_bars),
                "High": prices + abs(np.random.randn(n_bars)),
                "Low": prices - abs(np.random.randn(n_bars)),
                "Close": prices,
                "Volume": np.random.randint(100_000, 10_000_000, n_bars),
            },
            index=dates,
        )
    return data


@pytest.fixture
def sample_data():
    return _make_fake_data(n_tickers=3, n_bars=1000)


class TestDataSplitter:
    def test_split_ratios(self, sample_data):
        splits = DataSplitter.split(sample_data, 0.6, 0.2, 0.2)
        assert len(splits.train["TICK0"]) == 600
        assert len(splits.val["TICK0"]) == 200
        assert len(splits.test["TICK0"]) == 200

    def test_default_ratios(self, sample_data):
        splits = DataSplitter.split(sample_data)
        total = len(splits.train["TICK0"]) + len(splits.val["TICK0"]) + len(splits.test["TICK0"])
        assert total == 1000

    def test_no_temporal_overlap(self, sample_data):
        splits = DataSplitter.split(sample_data)
        assert DataSplitter.validate_no_overlap(splits) is True

    def test_no_temporal_overlap_fail_val_test(self, sample_data):
        splits = DataSplitter.split(sample_data)
        # manually break it
        splits.val["TICK0"] = pd.concat([splits.val["TICK0"], splits.test["TICK0"].iloc[:10]])
        with pytest.raises(ValueError):
            DataSplitter.validate_no_overlap(splits)

    def test_no_temporal_overlap_fail_train_val(self, sample_data):
        splits = DataSplitter.split(sample_data)
        # manually break it
        splits.train["TICK0"] = pd.concat([splits.train["TICK0"], splits.val["TICK0"].iloc[:10]])
        with pytest.raises(ValueError):
            DataSplitter.validate_no_overlap(splits)

    def test_all_tickers_present(self, sample_data):
        splits = DataSplitter.split(sample_data)
        for ticker in sample_data:
            assert ticker in splits.train
            assert ticker in splits.val
            assert ticker in splits.test

    def test_chronological_order(self, sample_data):
        splits = DataSplitter.split(sample_data)
        ref = "TICK0"
        train_max = splits.train[ref].index.max()
        val_min = splits.val[ref].index.min()
        val_max = splits.val[ref].index.max()
        test_min = splits.test[ref].index.min()
        assert train_max < val_min
        assert val_max < test_min

    def test_info_structure(self, sample_data):
        splits = DataSplitter.split(sample_data)
        info = splits.info
        assert "train" in info
        assert "val" in info
        assert "test" in info

    def test_invalid_ratios_sum(self, sample_data):
        with pytest.raises(ValueError):
            DataSplitter.split(sample_data, 0.5, 0.2, 0.2)

    def test_empty_data(self):
        with pytest.raises(ValueError):
            DataSplitter.split({})

    def test_too_few_bars(self):
        data = _make_fake_data(n_tickers=1, n_bars=5)
        with pytest.raises(ValueError):
            DataSplitter.split(data)

    def test_unequal_ticker_lengths(self):
        dates_long = pd.date_range("2023-01-01", periods=1000, freq="D")
        dates_short = pd.date_range("2023-01-01", periods=800, freq="D")
        data = {
            "LONG": pd.DataFrame({"Close": np.random.randn(1000)}, index=dates_long),
            "SHORT": pd.DataFrame({"Close": np.random.randn(800)}, index=dates_short),
        }
        splits = DataSplitter.split(data)
        assert splits.info["total_bars"] == 800

    def test_custom_ratios(self, sample_data):
        splits = DataSplitter.split(sample_data, 0.7, 0.15, 0.15)
        assert len(splits.train["TICK0"]) == 700
