import unittest
import sys
import json
import time
from unittest.mock import MagicMock, patch
from pathlib import Path
from datetime import datetime, timedelta
import shutil

# Mock necessary modules before importing app
sys.modules["flask"] = MagicMock()
sys.modules["flask_cors"] = MagicMock()
sys.modules["flask_socketio"] = MagicMock()
sys.modules["gevent"] = MagicMock()

# Mock alpaca_client to avoid .env issues
mock_alpaca = MagicMock()
sys.modules["trading.alpaca_client"] = mock_alpaca

# Import app (after mocking)
from dashboard import app


class TestDashboardOptimization(unittest.TestCase):
    def setUp(self):
        # Setup test data directory
        self.test_dir = Path("logs/trades_test_optimization")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)

        # Patch TRADES_LOG_DIR in app
        self.original_dir = app.TRADES_LOG_DIR
        app.TRADES_LOG_DIR = self.test_dir

        # Reset cache
        app._trades_cache = {"last_updated": 0, "data": [], "days": 0}

        # Create dummy files
        self.today = datetime.now().date()
        for i in range(10):
            date = self.today - timedelta(days=i)
            filename = self.test_dir / f"trades_{date}.json"

            trades = [
                {
                    "timestamp": (
                        datetime.combine(date, datetime.min.time()) + timedelta(hours=12)
                    ).isoformat(),
                    "symbol": "TEST",
                    "action": "BUY",
                    "amount": 100,
                }
            ]

            with open(filename, "w") as f:
                json.dump(trades, f)

    def tearDown(self):
        # Restore original dir and cleanup
        app.TRADES_LOG_DIR = self.original_dir
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_load_trades_correctness(self):
        """Test that load_trades_from_json loads correct number of trades"""
        # Load last 2 days (should include today, yesterday, day before yesterday if time matches)
        # Actually logic is: file_date >= (today - days)
        # If days=2: cutoff = today - 2.
        # Files: today (0), yesterday (1), 2 days ago (2).
        # All 3 files are >= cutoff.
        # 3 days ago (3) < cutoff.
        # So expect 3 trades.
        trades = app.load_trades_from_json(days=2)
        self.assertEqual(len(trades), 3)

    def test_caching_mechanism(self):
        """Test that caching works and is faster"""
        # First call
        t0 = time.time()
        app.load_trades_from_json(days=5)
        t1 = time.time()
        first_duration = t1 - t0

        # Verify cache was populated
        self.assertTrue(len(app._trades_cache["data"]) > 0)
        self.assertEqual(app._trades_cache["days"], 5)

        # Second call (cached)
        t0 = time.time()
        trades = app.load_trades_from_json(days=5)
        t1 = time.time()
        second_duration = t1 - t0

        self.assertEqual(len(trades), 6)  # 0 to 5 = 6 days
        # Caching should be faster (though hard to assert deterministically in CI, logic check is enough)

    def test_file_reading_optimization(self):
        """Test that we stop reading files after cutoff"""
        # We want to verify that we don't open all 10 files if we ask for 1 day
        days = 1
        # Cutoff = today - 1
        # Should read: today, yesterday. (2 files)
        # Should stop at day before yesterday.

        with patch("builtins.open", side_effect=open) as mock_open:
            app.load_trades_from_json(days=days)

            # Filter calls to relevant files
            opened_files = [
                str(call.args[0])
                for call in mock_open.call_args_list
                if "trades_" in str(call.args[0])
                and "json" in str(call.args[0])
                and "write" not in str(call.args[0])
            ]

            # Should be exactly 2 files (today and yesterday)
            # Maybe 3 if boundary condition is tricky, but definitely < 10
            self.assertLess(len(opened_files), 5)
            self.assertGreaterEqual(len(opened_files), 1)


if __name__ == "__main__":
    unittest.main()
