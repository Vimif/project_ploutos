import pytest
import logging
from unittest.mock import patch, MagicMock
from core.utils import format_duration, timestamp, setup_logging, cleanup_resources, get_gpu_info


def test_format_duration():
    assert format_duration(3661) == "01:01:01"
    assert format_duration(0) == "00:00:00"
    assert format_duration(59) == "00:00:59"


def test_timestamp():
    ts = timestamp()
    assert len(ts) == 15
    assert "_" in ts


def test_setup_logging():
    logger = setup_logging("test_logger")
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO


def test_setup_logging_with_file():
    with patch("logging.FileHandler"):
        logger = setup_logging("test_logger_file", log_file="test.log")
        assert logger.name == "test_logger_file"


@patch("gc.collect")
def test_cleanup_resources(mock_gc):
    mock_obj = MagicMock()
    cleanup_resources(mock_obj)
    mock_obj.close.assert_called_once()
    mock_gc.assert_called_once()


def test_cleanup_resources_close_error():
    mock_obj = MagicMock()
    mock_obj.close.side_effect = Exception("Close error")
    with patch("gc.collect"):
        cleanup_resources(mock_obj)
    mock_obj.close.assert_called_once()


def test_get_gpu_info_no_torch():
    with patch.dict("sys.modules", {"torch": None}):
        assert get_gpu_info() == {"available": False}


def test_get_gpu_info_with_torch():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_name.return_value = "Test GPU"
    mock_torch.cuda.memory_allocated.return_value = 1e9
    mock_torch.cuda.memory_reserved.return_value = 2e9
    mock_props = MagicMock()
    mock_props.total_memory = 8e9
    mock_torch.cuda.get_device_properties.return_value = mock_props

    with patch.dict("sys.modules", {"torch": mock_torch}):
        info = get_gpu_info()
        assert info["available"] is True
        assert info["name"] == "Test GPU"
        assert info["memory_allocated_gb"] == 1.0
        assert info["memory_reserved_gb"] == 2.0
        assert info["memory_total_gb"] == 8.0


def test_cleanup_resources_no_torch():
    mock_obj = MagicMock()
    with patch.dict("sys.modules", {"torch": None}):
        with patch("gc.collect"):
            cleanup_resources(mock_obj)
            mock_obj.close.assert_called_once()


def test_get_gpu_info_torch_no_cuda():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    with patch.dict("sys.modules", {"torch": mock_torch}):
        info = get_gpu_info()
        assert info["available"] is False


def test_get_gpu_info_import_error():
    import builtins

    original_import = builtins.__import__

    def side_effect(name, *args, **kwargs):
        if name == "torch":
            raise ImportError()
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=side_effect):
        assert get_gpu_info() == {"available": False}


def test_cleanup_resources_import_error():
    import builtins

    original_import = builtins.__import__

    def side_effect(name, *args, **kwargs):
        if name == "torch":
            raise ImportError()
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=side_effect):
        with patch("gc.collect"):
            cleanup_resources()


def test_cleanup_resources_torch_cuda():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    with patch.dict("sys.modules", {"torch": mock_torch}):
        with patch("gc.collect"):
            cleanup_resources()
    mock_torch.cuda.empty_cache.assert_called_once()
