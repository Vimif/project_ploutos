import json
import os
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock modules BEFORE importing dashboard.app
mock_flask_app_instance = MagicMock()
mock_flask_app_instance.config = {}

mock_flask = MagicMock()
mock_flask.Flask = MagicMock(return_value=mock_flask_app_instance)

mock_flask_cors = MagicMock()
mock_flask_socketio = MagicMock()
mock_gevent = MagicMock()
mock_trading_alpaca = MagicMock()

modules = {
    "flask": mock_flask,
    "flask_cors": mock_flask_cors,
    "flask_socketio": mock_flask_socketio,
    "gevent": mock_gevent,
    "trading.alpaca_client": mock_trading_alpaca,
    "trading": MagicMock(),
}

with patch.dict(sys.modules, modules):
    # Now import app
    import dashboard.app as app


class TestDashboardOptimization(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for trades
        self.test_dir = Path("tests/temp_trades_logs")
        self.test_dir.mkdir(parents=True, exist_ok=True)

        # Patch TRADES_LOG_DIR in the app module
        self.original_trades_dir = app.TRADES_LOG_DIR
        app.TRADES_LOG_DIR = self.test_dir

        # Reset cache if it exists (for future tests)
        if hasattr(app, "_trades_cache"):
            app._trades_cache = {"time": 0, "days": 0, "data": []}

    def tearDown(self):
        # Restore original directory
        app.TRADES_LOG_DIR = self.original_trades_dir

        # Clean up temp files
        if self.test_dir.exists():
            for f in self.test_dir.glob("*"):
                try:
                    f.unlink()
                except Exception:
                    pass
            try:
                self.test_dir.rmdir()
            except Exception:
                pass

    def create_dummy_trades(self, num_days=100):
        """Create dummy trade files for the last num_days"""
        base_date = datetime.now()
        files_created = []

        for i in range(num_days):
            date = base_date - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            filename = self.test_dir / f"trades_{date_str}.json"

            # Create a dummy trade
            trade_data = [
                {
                    "timestamp": date.isoformat(),
                    "symbol": "AAPL",
                    "action": "BUY",
                    "quantity": 1,
                    "price": 150.0,
                    "amount": 150.0,
                    "reason": "Test",
                    "portfolio_value": 10000 + i,
                }
            ]

            with open(filename, "w") as f:
                json.dump(trade_data, f)
            files_created.append(filename)

        return files_created

    def test_load_trades_performance(self):
        """Test how many files are opened when loading trades"""
        # Create 100 days of history
        self.create_dummy_trades(100)

        # We want to load only the last 5 days
        days_to_load = 5
        cutoff_date = datetime.now() - timedelta(days=days_to_load)

        # Mock open to count calls
        real_open = open
        open_mock = MagicMock(side_effect=real_open)

        # We patch builtins.open
        with patch("builtins.open", open_mock):
            trades = app.load_trades_from_json(days=days_to_load)

        # Verify we got trades
        self.assertTrue(len(trades) > 0)

        # Count how many JSON files were opened
        # We need to filter calls to open() that target our temp dir
        opened_files_count = 0
        for call in open_mock.mock_calls:
            if call[0] == "" or call[0] is None:  # This is the call itself, e.g. open(...)
                if call.args:
                    filepath = str(call.args[0])
                    if str(self.test_dir) in filepath and filepath.endswith(".json"):
                        opened_files_count += 1

        print(
            f"\n[Performance] Files opened: {opened_files_count} (for {days_to_load} days request out of 100 days history)"
        )

        # Currently, it opens ALL files. Optimizing this is the goal.
        # This assert documents the inefficiency if > 10
        if opened_files_count > 10:
            print("⚠️ Inefficient: Opening too many files!")
        else:
            print("✅ Efficient: Only opening necessary files.")

        # Verify correctness
        for trade in trades:
            trade_date = datetime.fromisoformat(trade["timestamp"])
            # Since load_trades_from_json filters strictly > cutoff, check that
            # Note: The original code uses strict > comparison
            self.assertTrue(
                trade_date > cutoff_date, f"Trade date {trade_date} should be > {cutoff_date}"
            )

    def test_cache_mechanism(self):
        """Test that caching prevents file reads"""
        self.create_dummy_trades(10)
        days = 5

        # First call - should open files
        real_open = open
        open_mock = MagicMock(side_effect=real_open)

        with patch("builtins.open", open_mock):
            app.load_trades_from_json(days=days)

            # Count opens
            first_call_opens = len(
                [
                    c
                    for c in open_mock.mock_calls
                    if c[0] == "" and c.args and str(self.test_dir) in str(c.args[0])
                ]
            )
            self.assertTrue(first_call_opens > 0, "First call should open files")

            # Reset mock
            open_mock.reset_mock()

            # Second call - should use cache
            app.load_trades_from_json(days=days)

            second_call_opens = len(
                [
                    c
                    for c in open_mock.mock_calls
                    if c[0] == "" and c.args and str(self.test_dir) in str(c.args[0])
                ]
            )
            print(f"\n[Cache] Files opened on second call: {second_call_opens}")
            self.assertEqual(second_call_opens, 0, "Second call should use cache and open 0 files")


if __name__ == "__main__":
    unittest.main()
