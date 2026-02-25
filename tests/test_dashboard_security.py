import base64
import os
import sys
import unittest
from unittest.mock import MagicMock, patch


class TestDashboardSecurity(unittest.TestCase):
    def setUp(self):
        # Setup environment variables needed for the app
        self.env_patcher = patch.dict(
            os.environ,
            {
                "DASHBOARD_USERNAME": "admin",
                "DASHBOARD_PASSWORD": "securepassword123",
                "FLASK_SECRET_KEY": "testing_key",
                "ALPACA_PAPER_API_KEY": "fake_key",
                "ALPACA_PAPER_SECRET_KEY": "fake_secret",
            },
        )
        self.env_patcher.start()

        # Mock dependencies modules to avoid import side effects or requirement issues
        # We use a dict to patch sys.modules temporarily
        self.modules_patcher = patch.dict(
            sys.modules,
            {
                "trading.alpaca_client": MagicMock(),
                "core.utils": MagicMock(),
            },
        )
        self.modules_patcher.start()

        # Mock specific components used by app
        sys.modules["core.utils"].setup_logging = MagicMock(return_value=MagicMock())

        # Setup mock Alpaca client class
        self.mock_alpaca_client_cls = MagicMock()
        self.mock_alpaca_instance = self.mock_alpaca_client_cls.return_value
        self.mock_alpaca_instance.get_account.return_value = {
            "portfolio_value": 100000.0,
            "cash": 50000.0,
            "buying_power": 200000.0,
            "equity": 100000.0,
            "last_equity": 99000.0,
        }
        sys.modules["trading.alpaca_client"].AlpacaClient = self.mock_alpaca_client_cls

        # Import the app inside the test method or setup to apply mocks
        # Since app is a module-level variable in dashboard.app, we need to be careful.
        # Ideally, we reload it, but standard import might work if it hasn't been imported yet.
        # Or we patch `dashboard.app.alpaca_client` after import.

        try:
            import dashboard.app

            self.app_module = dashboard.app
            # Force re-initialization or injection
            self.app_module.alpaca_client = self.mock_alpaca_instance
            self.app = self.app_module.app
            self.app.config["TESTING"] = True
            self.client = self.app.test_client()
        except ImportError:
            self.fail("Could not import dashboard.app")

    def tearDown(self):
        self.modules_patcher.stop()
        self.env_patcher.stop()

    def test_unauthorized_access(self):
        """Test that sensitive endpoints require authentication."""
        # Attempt to access a sensitive endpoint without auth
        response = self.client.get("/api/account")

        # Should return 401 Unauthorized
        self.assertEqual(
            response.status_code, 401, "Endpoint /api/account should require authentication"
        )
        self.assertIn("WWW-Authenticate", response.headers)

    def test_authorized_access(self):
        """Test that access is granted with correct credentials."""
        # Get the credentials we set in setUp
        username = "admin"
        password = "securepassword123"

        # Create Basic Auth header
        auth_str = f"{username}:{password}"
        auth_bytes = auth_str.encode("ascii")
        base64_bytes = base64.b64encode(auth_bytes)
        base64_str = base64_bytes.decode("ascii")
        headers = {"Authorization": f"Basic {base64_str}"}

        response = self.client.get("/api/account", headers=headers)

        # Should return 200 OK
        self.assertEqual(response.status_code, 200, "Should allow access with correct credentials")


if __name__ == "__main__":
    unittest.main()
