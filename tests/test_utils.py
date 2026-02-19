import numpy as np
import pandas as pd
import pytest

from core.data_pipeline import DataSplitter
from core.utils import cleanup_resources, format_duration, get_gpu_info, setup_logging, timestamp


def test_setup_logging():
    """Test logger initialization."""
    logger = setup_logging("test_logger")
    assert logger.name == "test_logger"
    assert logger.level > 0


def test_format_duration():
    """Test duration formatting."""
    assert format_duration(3661) == "01:01:01"
    assert format_duration(0) == "00:00:00"


def test_timestamp():
    """Test timestamp generation."""
    ts = timestamp()
    assert len(ts) == 15
    assert "_" in ts


def test_gpu_info():
    """Test GPU info (safe call even without GPU)."""
    info = get_gpu_info()
    assert isinstance(info, dict)
    assert "available" in info


def test_cleanup():
    """Test resource cleanup."""

    class MockResource:
        def close(self):
            pass

    res = MockResource()
    cleanup_resources(res)


def test_data_splitter_validation():
    """Test DataSplitter validation logic."""
    # Create valid data
    dates = pd.date_range(
        "2020-01-01", "2021-01-01", freq="h"
    )  # Using 'h' to match typical data frequency
    df = pd.DataFrame({"Close": np.random.randn(len(dates))}, index=dates)
    data = {"AAPL": df}

    # Test valid split
    splits = DataSplitter.split(data, 0.6, 0.2, 0.2)
    assert len(splits.train["AAPL"]) > 0
    assert len(splits.val["AAPL"]) > 0
    assert len(splits.test["AAPL"]) > 0

    # Test validation
    DataSplitter.validate_no_overlap(splits)


def test_data_splitter_edge_cases():
    """Test DataSplitter with edge cases."""
    # Test with empty dataframe
    with pytest.raises(ValueError):
        DataSplitter.split({}, 0.6, 0.2, 0.2)
