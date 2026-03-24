import pytest
import logging
import sys
from unittest.mock import patch, MagicMock
import core.utils as utils

def test_format_duration():
    assert utils.format_duration(3661.0) == "01:01:01"
    assert utils.format_duration(0.0) == "00:00:00"
    assert utils.format_duration(59.9) == "00:00:59"

def test_timestamp():
    ts = utils.timestamp()
    assert len(ts) == 15
    assert "_" in ts

@patch("core.utils.logging.getLogger")
def test_setup_logging(mock_get_logger):
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger

    logger = utils.setup_logging("test_logger")
    assert logger == mock_logger
    mock_logger.setLevel.assert_called_with(logging.INFO)

def test_get_gpu_info_unavailable():
    # If torch is imported, mock it to unavailable
    if "torch" in sys.modules:
        with patch("torch.cuda.is_available", return_value=False):
            info = utils.get_gpu_info()
            assert info['available'] is False
    else:
        # If not, let it fail naturally or simulate import error
        with patch.dict('sys.modules', {'torch': None}):
            info = utils.get_gpu_info()
            assert info['available'] is False

@patch("core.utils.gc.collect")
def test_cleanup_resources(mock_gc):
    class MockResource:
        def __init__(self):
            self.closed = False
        def close(self):
            self.closed = True

    res = MockResource()
    utils.cleanup_resources(res, None)
    assert res.closed is True
    mock_gc.assert_called_once()
