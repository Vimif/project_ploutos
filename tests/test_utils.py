import logging
from unittest.mock import MagicMock, patch

from core.utils import cleanup_resources, format_duration, get_gpu_info, setup_logging, timestamp


def test_setup_logging():
    logger = setup_logging("test_logger")
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
    assert len(logger.handlers) >= 1

    with patch("config.settings.LOGS_DIR", new_callable=MagicMock) as mock_logs_dir:
        mock_path = MagicMock()
        mock_logs_dir.__truediv__.return_value = mock_path

        # Test with file
        with patch("logging.FileHandler") as mock_fh:
            logger2 = setup_logging("test_logger2", "test.log")
            assert len(logger2.handlers) >= 1
            mock_fh.assert_called_once_with(mock_path)


def test_cleanup_resources():
    mock_obj = MagicMock()
    mock_obj.close = MagicMock()

    mock_obj_err = MagicMock()
    mock_obj_err.close = MagicMock(side_effect=Exception("error"))

    # Just run it to hit the lines
    with patch("gc.collect") as mock_gc:
        with patch.dict("sys.modules", {"torch": MagicMock()}):
            import torch

            torch.cuda.is_available.return_value = True
            cleanup_resources(mock_obj, mock_obj_err, None)

            mock_obj.close.assert_called_once()
            mock_obj_err.close.assert_called_once()
            torch.cuda.empty_cache.assert_called_once()
            mock_gc.assert_called_once()


def test_cleanup_resources_no_torch():
    with patch("gc.collect") as mock_gc:
        with patch.dict("sys.modules", {"torch": None}):
            cleanup_resources()
            mock_gc.assert_called_once()


def test_get_gpu_info():
    with patch.dict("sys.modules", {"torch": MagicMock()}):
        import torch

        torch.cuda.is_available.return_value = False
        info = get_gpu_info()
        assert info["available"] is False

        torch.cuda.is_available.return_value = True
        torch.cuda.get_device_name.return_value = "Test GPU"
        torch.cuda.memory_allocated.return_value = 1e9
        torch.cuda.memory_reserved.return_value = 2e9

        mock_props = MagicMock()
        mock_props.total_memory = 8e9
        torch.cuda.get_device_properties.return_value = mock_props

        info = get_gpu_info()
        assert info["available"] is True
        assert info["name"] == "Test GPU"
        assert info["memory_allocated_gb"] == 1.0
        assert info["memory_reserved_gb"] == 2.0
        assert info["memory_total_gb"] == 8.0


def test_get_gpu_info_no_torch():
    with patch.dict("sys.modules", {"torch": None}):
        info = get_gpu_info()
        assert info["available"] is False


def test_format_duration():
    assert format_duration(0) == "00:00:00"
    assert format_duration(61) == "00:01:01"
    assert format_duration(3665) == "01:01:05"


def test_timestamp():
    ts = timestamp()
    assert isinstance(ts, str)
    assert len(ts) == 15
