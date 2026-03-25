import pytest
import logging
from unittest.mock import MagicMock, patch
from core.utils import setup_logging, cleanup_resources, get_gpu_info, format_duration, timestamp

def test_setup_logging_no_file():
    logger = setup_logging("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
    assert len(logger.handlers) >= 1

def test_cleanup_resources():
    mock_obj = MagicMock()
    mock_obj.close = MagicMock()

    # Run with a valid object and None
    cleanup_resources(mock_obj, None)
    mock_obj.close.assert_called_once()

    # Run with object that throws on close
    mock_obj_err = MagicMock()
    mock_obj_err.close.side_effect = Exception("Boom")
    cleanup_resources(mock_obj_err)

def test_get_gpu_info_no_torch():
    with patch.dict('sys.modules', {'torch': None}):
        info = get_gpu_info()
        assert not info.get('available')

def test_get_gpu_info_no_cuda():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    with patch.dict('sys.modules', {'torch': mock_torch}):
        info = get_gpu_info()
        assert not info.get('available')

def test_get_gpu_info_with_cuda():
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_name.return_value = "Test GPU"
    mock_torch.cuda.memory_allocated.return_value = 1e9
    mock_torch.cuda.memory_reserved.return_value = 2e9
    mock_torch.cuda.get_device_properties.return_value.total_memory = 8e9

    with patch.dict('sys.modules', {'torch': mock_torch}):
        info = get_gpu_info()
        assert info['available']
        assert info['name'] == "Test GPU"
        assert info['memory_allocated_gb'] == 1.0
        assert info['memory_reserved_gb'] == 2.0
        assert info['memory_total_gb'] == 8.0

def test_format_duration():
    assert format_duration(0) == "00:00:00"
    assert format_duration(61) == "00:01:01"
    assert format_duration(3665) == "01:01:05"

def test_timestamp():
    ts = timestamp()
    assert len(ts) == 15
    assert "_" in ts
