from pathlib import Path

import pandas as pd

from core.data_fetcher import download_data


def _write_ohlcv_csv(path: Path, closes: list[float], *, freq: str) -> None:
    index = pd.date_range("2024-01-01", periods=len(closes), freq=freq)
    frame = pd.DataFrame(
        {
            "Open": closes,
            "High": [value + 1.0 for value in closes],
            "Low": [value - 1.0 for value in closes],
            "Close": closes,
            "Volume": [1000 + idx for idx in range(len(closes))],
        },
        index=index,
    )
    frame.to_csv(path)


def test_download_data_prefers_exact_local_interval_file(tmp_path):
    _write_ohlcv_csv(tmp_path / "SPY_1h.csv", [100.0, 101.0, 102.0, 103.0], freq="h")
    _write_ohlcv_csv(tmp_path / "SPY_4h.csv", [200.0, 205.0], freq="4h")

    result = download_data("SPY", interval="4h", dataset_path=str(tmp_path))

    assert list(result["Close"]) == [200.0, 205.0]
    assert len(result) == 2


def test_download_data_resamples_local_1h_to_4h_when_needed(tmp_path):
    _write_ohlcv_csv(
        tmp_path / "QQQ_1h.csv",
        [100.0, 102.0, 101.0, 104.0, 110.0, 111.0, 109.0, 112.0],
        freq="h",
    )

    result = download_data("QQQ", interval="4h", dataset_path=str(tmp_path))

    assert len(result) == 2
    assert result.iloc[0]["Open"] == 100.0
    assert result.iloc[0]["High"] == 105.0
    assert result.iloc[0]["Low"] == 99.0
    assert result.iloc[0]["Close"] == 104.0
    assert result.iloc[0]["Volume"] == 4006.0
    assert result.iloc[1]["Open"] == 110.0
    assert result.iloc[1]["High"] == 113.0
    assert result.iloc[1]["Low"] == 108.0
    assert result.iloc[1]["Close"] == 112.0
    assert result.iloc[1]["Volume"] == 4022.0
