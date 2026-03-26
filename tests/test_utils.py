# ruff: noqa: E402
import logging
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

from core.utils import cleanup_resources, format_duration, get_gpu_info, setup_logging, timestamp


def test_setup_logging_console_only():
    logger = setup_logging("test_logger_1")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger_1"
    assert logger.level == logging.INFO
    assert len(logger.handlers) >= 1
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)

@patch("config.settings.LOGS_DIR")
@patch("logging.FileHandler")
def test_setup_logging_with_file(mock_file_handler, mock_logs_dir):
    mock_path = MagicMock()
    mock_logs_dir.__truediv__.return_value = mock_path
    mock_instance = MagicMock()
    mock_file_handler.return_value = mock_instance

    logger = setup_logging("test_logger_2", log_file="test.log")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger_2"
    mock_logs_dir.__truediv__.assert_called_with("test.log")
    mock_file_handler.assert_called_with(mock_path)
    mock_instance.setFormatter.assert_called()

def test_cleanup_resources_with_closeable():
    mock_obj = MagicMock()
    cleanup_resources(mock_obj)
    mock_obj.close.assert_called_once()

def test_cleanup_resources_without_closeable():
    mock_obj = MagicMock(spec=[])
    cleanup_resources(mock_obj) # Should not raise error

@patch("gc.collect")
def test_cleanup_resources_none(mock_gc):
    cleanup_resources(None)
    mock_gc.assert_called_once()


@patch.dict(sys.modules, {"torch": MagicMock()})
def test_get_gpu_info_available():
    import sys
    mock_torch = sys.modules["torch"]
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_name.return_value = "Test GPU"
    mock_torch.cuda.memory_allocated.return_value = 1e9
    mock_torch.cuda.memory_reserved.return_value = 2e9
    mock_props = MagicMock()
    mock_props.total_memory = 8e9
    mock_torch.cuda.get_device_properties.return_value = mock_props

    info = get_gpu_info()

    assert info["available"] is True
    assert info["name"] == "Test GPU"
    assert info["memory_allocated_gb"] == 1.0
    assert info["memory_reserved_gb"] == 2.0
    assert info["memory_total_gb"] == 8.0

@patch.dict(sys.modules, {"torch": MagicMock()})
def test_get_gpu_info_unavailable():
    import sys
    mock_torch = sys.modules["torch"]
    mock_torch.cuda.is_available.return_value = False

    info = get_gpu_info()

    assert info["available"] is False
    assert "name" not in info

def test_format_duration():
    assert format_duration(0) == "00:00:00"
    assert format_duration(61) == "00:01:01"
    assert format_duration(3661) == "01:01:01"
    assert format_duration(3600 * 25 + 120 + 5) == "25:02:05"

@patch("core.utils.datetime")
def test_timestamp(mock_datetime):
    mock_now = datetime(2023, 10, 27, 14, 30, 0)
    mock_datetime.now.return_value = mock_now
    assert timestamp() == "20231027_143000"
