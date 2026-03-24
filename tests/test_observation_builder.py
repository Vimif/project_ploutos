import numpy as np
from core.observation_builder import ObservationBuilder


def test_observation_builder():
    n_assets = 2
    n_features = 3
    n_macro = 2

    tickers = ["A", "B"]
    feature_columns = ["f1", "f2", "f3"]
    feature_arrays = {
        "A": np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]], dtype=np.float32),
        "B": np.array([[4.0, 5.0, 6.0], [4.1, 5.1, 6.1]], dtype=np.float32),
    }
    macro_array = np.array([[7.0, 8.0], [7.1, 8.1]], dtype=np.float32)

    builder = ObservationBuilder(
        tickers=tickers,
        feature_columns=feature_columns,
        feature_arrays=feature_arrays,
        macro_array=macro_array,
        n_macro_features=n_macro,
    )

    portfolio = {"A": 10.0, "B": 20.0}
    prices = {"A": 100.0, "B": 200.0}
    equity = 10000.0
    balance = 5000.0
    initial_balance = 10000.0
    peak_value = 11000.0

    obs = builder.build(
        current_step=1,
        portfolio=portfolio,
        prices=prices,
        equity=equity,
        balance=balance,
        initial_balance=initial_balance,
        peak_value=peak_value,
    )

    assert len(obs) == builder.obs_size
    assert obs.shape == (13,)
    assert np.allclose(obs[:3], [1.1, 2.1, 3.1])
    assert np.allclose(obs[3:6], [4.1, 5.1, 6.1])
    assert np.allclose(obs[6:8], [7.1, 8.1])
    assert np.allclose(obs[8:10], [0.1, 0.4])
    assert np.allclose(obs[10], 0.5)
    assert np.allclose(obs[11], 0.0)
    assert np.allclose(obs[12], 0.0909090909)

    # Test edge cases where lengths are out of bounds
    obs_oob = builder.build(
        current_step=5,
        portfolio=portfolio,
        prices=prices,
        equity=equity,
        balance=balance,
        initial_balance=initial_balance,
        peak_value=peak_value,
    )

    assert np.allclose(obs_oob[:6], 0.0)
    assert np.allclose(obs_oob[6:8], 0.0)
