import os
import sys
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Ensure we can import dashboard.app
sys.path.insert(0, os.getcwd())

# We need to mock dependencies before importing dashboard.app
# because it does side effects on import (like connecting to Alpaca or setting up logging)
with (
    patch("dashboard.app.init_alpaca"),
    patch("dashboard.app.setup_logging"),
    patch("dashboard.app.SocketIO"),
    patch("dashboard.app.AlpacaClient"),
):
    from dashboard.app import CACHE_TTL, _trades_cache, load_trades_from_json


class TestDashboardOptimization(unittest.TestCase):

    def setUp(self):
        # Reset cache before each test
        _trades_cache["data"] = None
        _trades_cache["timestamp"] = 0
        _trades_cache["days"] = None

    @patch("dashboard.app.TRADES_LOG_DIR")
    def test_filename_filtering(self, mock_log_dir):
        """Test that files older than 'days' are skipped without opening"""

        # Setup mock files
        # We'll simulate 3 files:
        # 1. Very old (should be skipped)
        # 2. Inside range (should be read)
        # 3. Inside range (should be read)

        today = datetime.now().date()
        old_date = today - timedelta(days=100)
        recent_date1 = today - timedelta(days=5)
        recent_date2 = today - timedelta(days=1)

        file_old = MagicMock()
        file_old.name = f"trades_{old_date.strftime('%Y-%m-%d')}.json"

        file_recent1 = MagicMock()
        file_recent1.name = f"trades_{recent_date1.strftime('%Y-%m-%d')}.json"

        file_recent2 = MagicMock()
        file_recent2.name = f"trades_{recent_date2.strftime('%Y-%m-%d')}.json"

        # Make mocks comparable for sorted()
        # Since sorted() sorts by default in ascending order, we need __lt__
        # But we are sorting reverse=True, so technically we need __lt__ too.
        file_old.__lt__ = lambda self, other: self.name < other.name
        file_recent1.__lt__ = lambda self, other: self.name < other.name
        file_recent2.__lt__ = lambda self, other: self.name < other.name

        # Mock glob to return these files
        mock_log_dir.exists.return_value = True
        mock_log_dir.glob.return_value = [file_recent2, file_recent1, file_old]

        # Mock open to verify which files are opened
        # We need to handle context manager for open
        with patch("builtins.open", unittest.mock.mock_open(read_data="[]")) as mock_file:
            # Call function with days=30
            load_trades_from_json(days=30)

            # Check which files were opened
            # We expect file_recent1 and file_recent2 to be opened
            # file_old should NOT be opened

            opened_files = [c.args[0] for c in mock_file.call_args_list]

            self.assertIn(file_recent1, opened_files)
            self.assertIn(file_recent2, opened_files)
            self.assertNotIn(file_old, opened_files)

    @patch("dashboard.app.TRADES_LOG_DIR")
    def test_caching(self, mock_log_dir):
        """Test that subsequent calls use cache"""
        mock_log_dir.exists.return_value = True
        mock_log_dir.glob.return_value = []

        # Force cache clear just in case
        _trades_cache["data"] = None
        _trades_cache["days"] = None

        # First call - should access file system
        with patch("builtins.open", unittest.mock.mock_open(read_data="[]")):
            load_trades_from_json(days=30)
            self.assertEqual(mock_log_dir.glob.call_count, 1)

            # Second call - should use cache and NOT access file system
            load_trades_from_json(days=30)
            self.assertEqual(mock_log_dir.glob.call_count, 1)  # Count should remain 1

            # Call with different days - should invalidate cache and access file system
            load_trades_from_json(days=7)
            self.assertEqual(mock_log_dir.glob.call_count, 2)  # Count increases

    @patch("dashboard.app.TRADES_LOG_DIR")
    def test_cache_expiration(self, mock_log_dir):
        """Test that cache expires after TTL"""

        mock_log_dir.exists.return_value = True
        mock_log_dir.glob.return_value = []

        with patch("builtins.open", unittest.mock.mock_open(read_data="[]")):
            # First call
            load_trades_from_json(days=30)
            self.assertEqual(mock_log_dir.glob.call_count, 1)

            # Fast forward time beyond TTL
            # We need to patch time.time in dashboard.app module specifically because it was imported as 'import time'
            # and used as 'time.time()'. Wait, patching 'dashboard.app.time.time' works if time was imported.

            original_time = time.time()
            with patch("dashboard.app.time.time") as mock_time:
                mock_time.return_value = original_time + CACHE_TTL + 1.0

                # Call again - should expire and reload
                load_trades_from_json(days=30)
                self.assertEqual(mock_log_dir.glob.call_count, 2)


if __name__ == "__main__":
    unittest.main()
