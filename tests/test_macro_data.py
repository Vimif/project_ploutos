from unittest.mock import patch

import pandas as pd
import pytest

from core.macro_data import MacroDataFetcher


@pytest.fixture
def mock_yfinance():
    # Patch yfinance inside fetch_all where it's imported
    with patch("yfinance.download") as mock:
        yield mock


def test_fetch_all(mock_yfinance):
    fetcher = MacroDataFetcher()

    # Mock download response
    mock_df = pd.DataFrame(
        {"Close": [10, 11, 12]}, index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    )

    mock_yfinance.return_value = mock_df

    data = fetcher.fetch_all()
    assert isinstance(data, pd.DataFrame)
