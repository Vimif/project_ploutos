import pytest
from unittest.mock import patch, MagicMock
import logging
from core.utils import format_duration, timestamp, get_gpu_info, cleanup_resources, setup_logging


def test_format_duration():
    assert format_duration(3600) == "01:00:00"
    assert format_duration(3661) == "01:01:01"
    assert format_duration(65) == "00:01:05"
    assert format_duration(0) == "00:00:00"


def test_timestamp():
    ts = timestamp()
    assert isinstance(ts, str)
    assert len(ts) == 15
    assert "_" in ts


def test_get_gpu_info_no_torch():
    with patch.dict("sys.modules", {"torch": None}):
        info = get_gpu_info()
        assert not info["available"]


def test_get_gpu_info_no_cuda():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    with patch.dict("sys.modules", {"torch": mock_torch}):
        info = get_gpu_info()
        assert not info["available"]


def test_get_gpu_info_with_cuda():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_name.return_value = "Mock GPU"
    mock_torch.cuda.memory_allocated.return_value = 1e9
    mock_torch.cuda.memory_reserved.return_value = 2e9

    mock_props = MagicMock()
    mock_props.total_memory = 8e9
    mock_torch.cuda.get_device_properties.return_value = mock_props

    with patch.dict("sys.modules", {"torch": mock_torch}):
        info = get_gpu_info()
        assert info["available"]
        assert info["name"] == "Mock GPU"
        assert info["memory_allocated_gb"] == 1.0
        assert info["memory_reserved_gb"] == 2.0
        assert info["memory_total_gb"] == 8.0


def test_cleanup_resources():
    # Test closeable object
    mock_obj = MagicMock()
    cleanup_resources(mock_obj)
    mock_obj.close.assert_called_once()

    # Test uncloseable object
    mock_obj2 = "Not closeable"
    cleanup_resources(mock_obj2)  # Should not raise

    # Test torch cleanup
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    with patch.dict("sys.modules", {"torch": mock_torch}):
        cleanup_resources()
        mock_torch.cuda.empty_cache.assert_called_once()


def test_setup_logging():
    logger = setup_logging("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
