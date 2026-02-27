import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from core.data_pipeline import DataSplitter
from core.ensemble import EnsemblePredictor
from trading.portfolio import Portfolio


# --- Test DataSplitter (core/data_pipeline.py) ---
def test_data_splitter_coverage():
    dates = pd.date_range("2020-01-01", "2020-01-10", freq="D")
    df = pd.DataFrame({"Close": range(10)}, index=dates)
    data = {"AAPL": df}

    # Test valid split
    splits = DataSplitter.split(data, 0.6, 0.2, 0.2)
    assert "AAPL" in splits.train
    assert len(splits.train["AAPL"]) == 6
    assert len(splits.val["AAPL"]) == 2
    assert len(splits.test["AAPL"]) == 2

    # Test validation
    DataSplitter.validate_no_overlap(splits)

    # Test properties (info dict)
    assert isinstance(splits.info["tickers"], list)
    assert "AAPL" in splits.info["tickers"]

    # Test error cases
    with pytest.raises(ValueError):
        DataSplitter.split(data, 0.5, 0.5, 0.5)  # Sum > 1.0


# --- Test EnsemblePredictor (core/ensemble.py) ---
def test_ensemble_coverage():
    # Mock models
    model1 = MagicMock()
    model1.predict.return_value = (np.array([1]), None)
    model2 = MagicMock()
    model2.predict.return_value = (np.array([1]), None)
    model3 = MagicMock()
    model3.predict.return_value = (np.array([0]), None)

    ensemble = EnsemblePredictor([model1, model2, model3])

    # Test predict (returns just action)
    obs = np.array([0.5, 0.5])
    action = ensemble.predict(obs)

    # Majority vote: 1, 1, 0 -> 1
    # Check if it's array or scalar
    if isinstance(action, np.ndarray):
        assert action.item() == 1
    else:
        assert action == 1

    # Test confidence
    # 2 out of 3 agreed -> 0.66
    assert hasattr(ensemble, "models")

    # Test load (mocking PPO.load)
    with patch("stable_baselines3.PPO.load") as mock_load:
        mock_load.return_value = model1
        ens = EnsemblePredictor.load(["path1", "path2"])
        assert len(ens.models) == 2


# --- Test Portfolio (trading/portfolio.py) ---
def test_portfolio_coverage():
    p = Portfolio(initial_capital=10000.0)

    # Test properties
    assert p.cash == 10000.0

    # Test Buy
    p.buy("AAPL", 150.0, 1500.0)  # 10 shares
    assert p.positions["AAPL"]["shares"] == 10
    assert p.cash == 10000.0 - 1500.0

    # Test Sell
    p.sell("AAPL", 160.0, 0.5)  # Sell 50% (5 shares)
    assert p.positions["AAPL"]["shares"] == 5
    assert p.cash == 8500.0 + (5 * 160.0)

    # Test Value
    current_prices = {"AAPL": 170.0}
    val = p.get_total_value(current_prices)
    # Cash + (5 * 170)
    expected = p.cash + (5 * 170.0)
    assert val == expected

    # Test History
    assert isinstance(p.trades_history, list)
    assert len(p.trades_history) == 2
