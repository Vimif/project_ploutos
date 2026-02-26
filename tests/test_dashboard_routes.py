import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDashboardRoutes(unittest.TestCase):
    def setUp(self):
        # Create a patcher for sys.modules
        self.modules_patcher = patch.dict(
            sys.modules,
            {
                "flask_socketio": MagicMock(),
                "gevent": MagicMock(),
                "trading.alpaca_client": MagicMock(),
                "core.utils": MagicMock(),
            },
        )
        self.modules_patcher.start()

        # Now import app.py
        # We need to force reload or just import for the first time
        if "dashboard.app" in sys.modules:
            del sys.modules["dashboard.app"]
        import dashboard.app

        self.app = dashboard.app.app
        self.app.config["TESTING"] = True
        self.app.config["SECRET_KEY"] = "test_secret"
        self.client = self.app.test_client()

    def tearDown(self):
        self.modules_patcher.stop()

    def test_index_route(self):
        rv = self.client.get("/")
        self.assertEqual(rv.status_code, 200)
        self.assertIn(b"Ploutos Dashboard", rv.data)

    def test_trades_route(self):
        rv = self.client.get("/trades")
        self.assertEqual(rv.status_code, 200, "Route /trades should exist")
        self.assertIn(b"Historique des Trades", rv.data)
        self.assertIn(b'id="btn-filter"', rv.data)
        self.assertIn(b'aria-live="polite"', rv.data)

    def test_metrics_route(self):
        rv = self.client.get("/metrics")
        self.assertEqual(rv.status_code, 200, "Route /metrics should exist")


if __name__ == "__main__":
    unittest.main()
