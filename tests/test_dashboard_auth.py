import base64
import os
import sys
import unittest
from unittest.mock import MagicMock

# Mock environment variables before importing anything
os.environ["DASHBOARD_PASSWORD"] = "secure_test_password"
os.environ["ALPACA_PAPER_API_KEY"] = "fake_key"
os.environ["ALPACA_PAPER_SECRET_KEY"] = "fake_secret"

# Mock AlpacaClient before importing app
sys.modules["trading.alpaca_client"] = MagicMock()
mock_alpaca = MagicMock()
sys.modules["trading.alpaca_client"].AlpacaClient.return_value = mock_alpaca

# Mock account data
mock_alpaca.get_account.return_value = {
    "portfolio_value": "100000",
    "cash": "50000",
    "buying_power": "200000",
    "equity": "100000",
    "last_equity": "99000",
}
mock_alpaca.get_positions.return_value = []
mock_alpaca.close_position.return_value = True

from dashboard.app import app  # noqa: E402


class TestDashboardAuth(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.password = "secure_test_password"
        # Ensure testing environment
        app.config["TESTING"] = True

    def test_unauthenticated_access(self):
        """Test access without credentials returns 401"""
        response = self.client.get("/api/account")
        self.assertEqual(response.status_code, 401)
        self.assertIn("Authentication required", response.data.decode())

    def test_wrong_password(self):
        """Test access with wrong credentials returns 401"""
        auth_str = base64.b64encode(b"admin:wrongpass").decode("utf-8")
        headers = {"Authorization": f"Basic {auth_str}"}
        response = self.client.get("/api/account", headers=headers)
        self.assertEqual(response.status_code, 401)

    def test_correct_password(self):
        """Test access with correct credentials returns 200"""
        auth_str = base64.b64encode(f"admin:{self.password}".encode()).decode("utf-8")
        headers = {"Authorization": f"Basic {auth_str}"}
        response = self.client.get("/api/account", headers=headers)
        self.assertEqual(response.status_code, 200)

    def test_close_position_auth(self):
        """Test sensitive action with auth"""
        auth_str = base64.b64encode(f"admin:{self.password}".encode()).decode("utf-8")
        headers = {"Authorization": f"Basic {auth_str}"}
        response = self.client.post("/api/close_position/AAPL", headers=headers)
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
