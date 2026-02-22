import sys
import os
import unittest
import base64
import importlib
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDashboardSecurity(unittest.TestCase):
    def setUp(self):
        # 1. Patch sys.modules to mock dependencies BEFORE importing app
        # We use patch.dict so it automatically restores original sys.modules on tearDown
        self.modules_patcher = patch.dict(
            sys.modules,
            {
                "flask_socketio": MagicMock(),
                "gevent": MagicMock(),
                # We mock the module where AlpacaClient is defined
                "trading.alpaca_client": MagicMock(),
            },
        )
        self.modules_patcher.start()

        # 2. Patch environment variables
        self.env_patcher = patch.dict(
            os.environ, {"DASHBOARD_USERNAME": "testuser", "DASHBOARD_PASSWORD": "testpass"}
        )
        self.env_patcher.start()

        # 3. Import the app inside the patched environment
        # If it was already imported, reload it to pick up the new env vars and mocks
        import dashboard.app

        importlib.reload(dashboard.app)

        self.app_module = dashboard.app
        self.app = self.app_module.app
        self.app.testing = True
        self.client = self.app.test_client()

        # 4. Setup specific mock behaviors for this test
        # Mock init_alpaca to avoid network calls
        self.alpaca_init_patch = patch("dashboard.app.init_alpaca", return_value=True)
        self.alpaca_init_patch.start()

        # Mock the global alpaca_client in dashboard.app
        self.app_module.alpaca_client = MagicMock()
        self.app_module.alpaca_client.get_account.return_value = {
            "portfolio_value": 100000,
            "cash": 50000,
            "buying_power": 200000,
            "equity": 100000,
            "last_equity": 99000,
        }

    def tearDown(self):
        self.alpaca_init_patch.stop()
        self.env_patcher.stop()
        self.modules_patcher.stop()

        # Clean up dashboard.app from sys.modules to ensure isolation for other tests
        # This prevents the "mocked" version of the app (and its dependencies) from leaking
        if "dashboard.app" in sys.modules:
            del sys.modules["dashboard.app"]

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
        self.app_module.alpaca_client.close_position.return_value = True
        headers = self.get_auth_headers("testuser", "testpass")
        response = self.client.post("/api/close_position/AAPL", headers=headers)
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
