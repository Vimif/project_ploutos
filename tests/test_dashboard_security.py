import base64
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# We need to set env vars before importing app if we want to control them at module level
# This ensures that when dashboard.app is imported, it picks up these values
os.environ["DASHBOARD_USERNAME"] = "testuser"
os.environ["DASHBOARD_PASSWORD"] = "testpass"

# Mock heavy dependencies or those that might fail in test env
# We mock them in sys.modules so that when dashboard.app imports them, it gets mocks
sys.modules["flask_socketio"] = MagicMock()
sys.modules["gevent"] = MagicMock()
# Note: we don't strictly need to mock alpaca_client module if we have it installed,
# but it's safer to avoid side effects.
# However, dashboard.app imports AlpacaClient from trading.alpaca_client
# If we mock the module, we need to make sure the class is there.
mock_alpaca_module = MagicMock()
sys.modules["trading.alpaca_client"] = mock_alpaca_module

# Now import the app
try:
    from dashboard import app as dashboard_app
except ImportError:
    # Fallback if imports fail (e.g. core.utils issues)
    # But since we installed deps, it should be fine.
    # We might need to mock core.utils too if it has side effects
    pass


class TestDashboardSecurity(unittest.TestCase):
    def setUp(self):
        # Setup the app for testing
        self.app = dashboard_app.app
        self.app.testing = True
        self.client = self.app.test_client()

        # Ensure we use the test credentials
        # Even though we set env vars, checking if they were picked up
        # If not (because import happened earlier), we patch them
        self.patcher_user = patch("dashboard.app.DASHBOARD_USERNAME", "testuser")
        self.patcher_pass = patch("dashboard.app.DASHBOARD_PASSWORD", "testpass")
        self.patcher_user.start()
        self.patcher_pass.start()

        # Mock init_alpaca to avoid network calls and return True
        self.alpaca_init_patch = patch("dashboard.app.init_alpaca", return_value=True)
        self.alpaca_init_patch.start()

        # Mock the global alpaca_client in dashboard.app
        dashboard_app.alpaca_client = MagicMock()
        dashboard_app.alpaca_client.get_account.return_value = {
            "portfolio_value": 100000,
            "cash": 50000,
            "buying_power": 200000,
            "equity": 100000,
            "last_equity": 99000,
        }

    def tearDown(self):
        self.patcher_user.stop()
        self.patcher_pass.stop()
        self.alpaca_init_patch.stop()

    def get_auth_headers(self, username, password):
        return {
            "Authorization": "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()
        }

    def test_unauthorized_access(self):
        """Test accessing a protected route without credentials."""
        response = self.client.get("/api/account")
        self.assertEqual(response.status_code, 401)
        self.assertIn("WWW-Authenticate", response.headers)
        self.assertIn('Basic realm="Login Required"', response.headers["WWW-Authenticate"])

    def test_wrong_credentials(self):
        """Test accessing a protected route with wrong credentials."""
        headers = self.get_auth_headers("testuser", "wrongpass")
        response = self.client.get("/api/account", headers=headers)
        self.assertEqual(response.status_code, 401)

    def test_correct_credentials(self):
        """Test accessing a protected route with correct credentials."""
        headers = self.get_auth_headers("testuser", "testpass")
        response = self.client.get("/api/account", headers=headers)
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["portfolio_value"], 100000)

    def test_root_protected(self):
        """Test that the root route (HTML) is also protected."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 401)

        headers = self.get_auth_headers("testuser", "testpass")
        response = self.client.get("/", headers=headers)
        self.assertEqual(response.status_code, 200)

    def test_close_position_protected(self):
        """Test that critical action close_position is protected."""
        response = self.client.post("/api/close_position/AAPL")
        self.assertEqual(response.status_code, 401)

        # With auth
        dashboard_app.alpaca_client.close_position.return_value = True
        headers = self.get_auth_headers("testuser", "testpass")
        response = self.client.post("/api/close_position/AAPL", headers=headers)
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
