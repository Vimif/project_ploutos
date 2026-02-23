import pytest
from unittest.mock import MagicMock, patch
import sys
from core.utils import setup_logging, cleanup_resources, get_gpu_info, format_duration, timestamp

class TestUtils:
    def test_setup_logging(self):
        logger = setup_logging("test_logger")
        assert logger.name == "test_logger"
        assert len(logger.handlers) > 0

    @patch('core.utils.logging.FileHandler')
    @patch('config.settings.LOGS_DIR')
    def test_setup_logging_with_file(self, mock_logs_dir, mock_file_handler):
        mock_logs_dir.__truediv__.return_value = "dummy.log"
        setup_logging("test_file_logger", log_file="test.log")
        assert mock_file_handler.called

    def test_cleanup_resources(self):
        obj = MagicMock()
        cleanup_resources(obj)
        obj.close.assert_called_once()

        # Test with object raising exception on close
        obj_fail = MagicMock()
        obj_fail.close.side_effect = Exception("error")
        cleanup_resources(obj_fail) # Should not raise

    def test_format_duration(self):
        assert format_duration(3661) == "01:01:01"
        assert format_duration(65) == "00:01:05"
        assert format_duration(10) == "00:00:10"

    def test_timestamp(self):
        ts = timestamp()
        assert len(ts) == 15 # YYYYMMDD_HHMMSS
        assert "_" in ts

    def test_get_gpu_info_no_torch(self):
        # Simulate import error
        with patch.dict(sys.modules, {'torch': None}):
            info = get_gpu_info()
            assert not info['available']

    def test_get_gpu_info_cuda(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Test GPU"
        mock_torch.cuda.memory_allocated.return_value = 1000000000
        mock_torch.cuda.memory_reserved.return_value = 2000000000
        mock_torch.cuda.get_device_properties.return_value.total_memory = 8000000000

        with patch.dict(sys.modules, {'torch': mock_torch}):
            info = get_gpu_info()
            assert info['available']
            assert info['name'] == "Test GPU"
            assert info['memory_allocated_gb'] == 1.0

    def test_get_gpu_info_no_cuda(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict(sys.modules, {'torch': mock_torch}):
            info = get_gpu_info()
            assert not info['available']
