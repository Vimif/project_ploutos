import logging
import pytest
from unittest.mock import MagicMock, patch

from core.utils import setup_logging, cleanup_resources, get_gpu_info, format_duration, timestamp

def test_setup_logging():
    # Test setting up a logger
    logger = setup_logging("test_logger")
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
    assert len(logger.handlers) >= 1

def test_format_duration():
    # Test basic duration formatting
    assert format_duration(0) == "00:00:00"
    assert format_duration(61) == "00:01:01"
    assert format_duration(3600) == "01:00:00"
    assert format_duration(3665) == "01:01:05"

def test_timestamp():
    # Test format matches exactly what is expected: '%Y%m%d_%H%M%S'
    ts = timestamp()
    assert len(ts) == 15
    assert "_" in ts

@patch("core.utils.gc.collect")
def test_cleanup_resources(mock_gc):
    # Test basic cleanup of objects
    mock_obj = MagicMock()
    cleanup_resources(mock_obj)
    mock_obj.close.assert_called_once()
    mock_gc.assert_called_once()

    # Test ignore errors during close
    mock_obj2 = MagicMock()
    mock_obj2.close.side_effect = Exception("Failed")
    cleanup_resources(mock_obj2)
    mock_obj2.close.assert_called_once()

def test_get_gpu_info_no_torch():
    # Test GPU info when torch is missing or not available
    with patch.dict('sys.modules', {'torch': None}):
        info = get_gpu_info()
        assert info == {'available': False}